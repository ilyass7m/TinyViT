# --------------------------------------------------------
# TinyViT Main (train/validate)
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Add distillation with saved teacher logits
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
from collections import defaultdict
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from my_meter import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint,\
    NativeScalerWithGradNormCount,\
    auto_resume_helper, is_main_process,\
    add_common_args,\
    get_git_info

from models.remap_layer import RemapLayer
remap_layer_22kto1k = RemapLayer('./imagenet_1kto22k.txt')

try:
    import wandb
except ImportError:
    wandb = None
NORM_ITER_LEN = 100


def parse_option():
    parser = argparse.ArgumentParser(
        'TinyViT training and evaluation script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(args, config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if not args.only_cpu:
        model.cuda()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Freeze early stages for faster finetuning
    freeze_stages = config.TRAIN.FREEZE_STAGES
    if freeze_stages > 0 and hasattr(model, 'patch_embed'):
        frozen_params = 0
        # Freeze patch_embed
        for param in model.patch_embed.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        # Freeze specified number of stages
        for i in range(min(freeze_stages, len(model.layers))):
            for param in model.layers[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Frozen {freeze_stages} stages: {frozen_params:,} / {total_params:,} params "
                    f"({100*frozen_params/total_params:.1f}%) - Only training {total_params-frozen_params:,} params")

    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # torch.compile() for PyTorch 2.0+ - can give 10-30% speedup
    if hasattr(torch, 'compile') and config.get('COMPILE', False):
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    loss_scaler = NativeScalerWithGradNormCount(grad_scaler_enabled=config.AMP_ENABLE)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(
        data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)

    # Online distillation setup (teacher model for live forward passes)
    teacher_model = None
    online_distill_loss = None
    feature_criterion = None

    if config.DISTILL.ONLINE_DISTILL:
        # Fully online distillation: teacher forward pass each batch
        logger.info("Online distillation enabled - loading teacher model...")
        from feature_distill import build_teacher_for_feature_distill, build_online_distill_loss
        teacher_model = build_teacher_for_feature_distill(config)
        if not args.only_cpu:
            teacher_model.cuda()
        teacher_model.eval()
        online_distill_loss = build_online_distill_loss(config)
        if not args.only_cpu:
            online_distill_loss.cuda()
        logger.info(f"Online distillation - Feature enabled: {config.DISTILL.FEATURE_ENABLED}")
        if config.DISTILL.FEATURE_ENABLED:
            logger.info(f"Feature distillation weight (beta): {config.DISTILL.FEATURE_WEIGHT}")
            logger.info(f"Student dim: {config.DISTILL.FEATURE_DIM_STUDENT}, Teacher dim: {config.DISTILL.FEATURE_DIM_TEACHER}")

    elif config.DISTILL.FEATURE_ENABLED:
        # Hybrid mode: saved logits + online features (existing functionality)
        logger.info("Feature distillation enabled (with saved logits) - loading teacher model...")
        from feature_distill import FeatureDistillationLoss, build_teacher_for_feature_distill
        teacher_model = build_teacher_for_feature_distill(config)
        if not args.only_cpu:
            teacher_model.cuda()
        teacher_model.eval()
        feature_criterion = FeatureDistillationLoss(
            student_dim=config.DISTILL.FEATURE_DIM_STUDENT,
            teacher_dim=config.DISTILL.FEATURE_DIM_TEACHER,
        )
        if not args.only_cpu:
            feature_criterion.cuda()
        logger.info(f"Feature distillation weight: {config.DISTILL.FEATURE_WEIGHT}")

    if config.DISTILL.ENABLED and not config.DISTILL.ONLINE_DISTILL:
        # Saved logits distillation requires TEACHER_LOGITS_PATH
        assert len(
            config.DISTILL.TEACHER_LOGITS_PATH) > 0, "Please fill in DISTILL.TEACHER_LOGITS_PATH"
        criterion = SoftTargetCrossEntropy()
    elif config.DISTILL.ONLINE_DISTILL:
        # Online distillation uses OnlineDistillationLoss, no need for SoftTargetCrossEntropy
        criterion = None  # Loss is computed in train_one_epoch_online_distill
    else:
        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(
                smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # set_epoch for dataset_train when distillation
        if hasattr(dataset_train, 'set_epoch'):
            dataset_train.set_epoch(epoch)
        data_loader_train.sampler.set_epoch(epoch)

        if config.DISTILL.ONLINE_DISTILL:
            # Fully online distillation: live teacher forward each batch
            # Supports both logits-only and logits+features modes via FEATURE_ENABLED
            train_one_epoch_online_distill(
                args, config, model, teacher_model, online_distill_loss,
                data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        elif config.DISTILL.ENABLED:
            if config.DISTILL.FEATURE_ENABLED:
                # Hybrid: Saved logits + Online features
                train_one_epoch_distill_with_features(
                    args, config, model, teacher_model, criterion, feature_criterion,
                    data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
            else:
                # Saved logits only (original TinyViT approach)
                train_one_epoch_distill_using_saved_logits(
                    args, config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        else:
            # Standard training (no distillation)
            train_one_epoch(args, config, model, criterion,
                            data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp,
                            max_accuracy, optimizer, lr_scheduler, loss_scaler, logger)

        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        if is_main_process() and args.use_wandb:
            wandb.log({
                f"val/acc@1": acc1,
                f"val/acc@5": acc5,
                f"val/loss": loss,
                "epoch": epoch,
            })
            wandb.run.summary['epoch'] = epoch
            wandb.run.summary['best_acc@1'] = max_accuracy

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def is_valid_grad_norm(num):
    if num is None:
        return False
    return not bool(torch.isinf(num)) and not bool(torch.isnan(num))


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def train_one_epoch(args, config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets

        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))

        # Note: removed torch.cuda.synchronize() - it was blocking every iteration
        loss_meter.update(loss.item(), targets.size(0))
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            if is_main_process() and args.use_wandb:
                wandb.log({
                    "train/acc@1": acc1_meter.val,
                    "train/acc@5": acc5_meter.val,
                    "train/loss": loss_meter.val,
                    "train/grad_norm": norm_meter.val,
                    "train/loss_scale": scaler_meter.val,
                    "train/lr": lr,
                }, step=normal_global_idx)
    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train_one_epoch_distill_using_saved_logits(args, config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    data_tic = time.time()

    num_classes = config.MODEL.NUM_CLASSES
    topk = config.DISTILL.LOGITS_TOPK

    for idx, ((samples, targets), (logits_index, logits_value, seeds)) in enumerate(data_loader):
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, seeds)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets
        meters['data_time'].update(time.time() - data_tic)

        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            outputs = model(samples)

        # recover teacher logits
        logits_index = logits_index.long()
        logits_value = logits_value.float()
        logits_index = logits_index.cuda(non_blocking=True)
        logits_value = logits_value.cuda(non_blocking=True)

        # Handle case where topk >= num_classes (e.g., CIFAR-100 with topk=100)
        if topk >= num_classes:
            # All classes are in top-k, no need for minor_value distribution
            # Just use the saved logits directly (they cover all classes)
            outputs_teacher = torch.zeros(logits_value.size(0), num_classes,
                                         device=logits_value.device, dtype=logits_value.dtype)
            outputs_teacher = outputs_teacher.scatter_(-1, logits_index, logits_value)
        else:
            # Original sparse reconstruction for large num_classes (e.g., ImageNet-22k)
            minor_value = (1.0 - logits_value.sum(-1, keepdim=True)
                           ) / (num_classes - topk)
            minor_value = minor_value.repeat_interleave(num_classes, dim=-1)
            outputs_teacher = minor_value.scatter_(-1, logits_index, logits_value)

        loss = criterion(outputs, outputs_teacher)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        # compute accuracy
        real_batch_size = len(original_targets)
        acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        meters['train_acc1'].update(acc1.item(), real_batch_size)
        meters['train_acc5'].update(acc5.item(), real_batch_size)
        teacher_acc1, teacher_acc5 = accuracy(
            outputs_teacher, original_targets, topk=(1, 5))
        meters['teacher_acc1'].update(teacher_acc1.item(), real_batch_size)
        meters['teacher_acc5'].update(teacher_acc5.item(), real_batch_size)

        # Note: removed torch.cuda.synchronize() - it was blocking every iteration
        loss_meter.update(loss.item(), real_batch_size)
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        data_tic = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

            if is_main_process() and args.use_wandb:
                acc1_meter, acc5_meter = meters['train_acc1'], meters['train_acc5']
                wandb.log({
                    "train/acc@1": acc1_meter.val,
                    "train/acc@5": acc5_meter.val,
                    "train/loss": loss_meter.val,
                    "train/grad_norm": norm_meter.val,
                    "train/loss_scale": scaler_meter.val,
                    "train/lr": lr,
                }, step=normal_global_idx)
    epoch_time = time.time() - start
    extra_meters_str = f'Train-Summary: [{epoch}/{config.TRAIN.EPOCHS}]\t'
    for k, v in meters.items():
        v.sync()
        extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
    logger.info(extra_meters_str)
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train_one_epoch_distill_with_features(
    args, config, model, teacher_model, criterion, feature_criterion,
    data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler
):
    """
    Training with online feature distillation.

    Uses saved logits for logit distillation + live teacher for feature distillation.
    Teacher model runs in eval mode with no gradients.
    """
    from feature_distill import get_student_features

    model.train()
    teacher_model.eval()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    data_tic = time.time()

    num_classes = config.MODEL.NUM_CLASSES
    topk = config.DISTILL.LOGITS_TOPK
    feature_weight = config.DISTILL.FEATURE_WEIGHT

    for idx, ((samples, targets), (logits_index, logits_value, seeds)) in enumerate(data_loader):
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, seeds)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets
        meters['data_time'].update(time.time() - data_tic)

        # Get teacher features (no gradient)
        with torch.no_grad():
            _, teacher_features = teacher_model(samples)

        # Get student outputs and features
        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            outputs, student_features = get_student_features(model, samples)

        # Recover teacher logits from saved data
        logits_index = logits_index.long()
        logits_value = logits_value.float()
        logits_index = logits_index.cuda(non_blocking=True)
        logits_value = logits_value.cuda(non_blocking=True)

        if topk >= num_classes:
            outputs_teacher = torch.zeros(logits_value.size(0), num_classes,
                                         device=logits_value.device, dtype=logits_value.dtype)
            outputs_teacher = outputs_teacher.scatter_(-1, logits_index, logits_value)
        else:
            minor_value = (1.0 - logits_value.sum(-1, keepdim=True)) / (num_classes - topk)
            minor_value = minor_value.repeat_interleave(num_classes, dim=-1)
            outputs_teacher = minor_value.scatter_(-1, logits_index, logits_value)

        # Logit distillation loss (using saved logits)
        logit_loss = criterion(outputs, outputs_teacher)

        # Feature distillation loss (using live teacher features)
        feature_loss = feature_criterion(student_features.float(), teacher_features.float())

        # Combined loss
        loss = logit_loss + feature_weight * feature_loss
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # Backward pass
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        # Compute accuracy
        real_batch_size = len(original_targets)
        acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        meters['train_acc1'].update(acc1.item(), real_batch_size)
        meters['train_acc5'].update(acc5.item(), real_batch_size)
        teacher_acc1, teacher_acc5 = accuracy(
            outputs_teacher, original_targets, topk=(1, 5))
        meters['teacher_acc1'].update(teacher_acc1.item(), real_batch_size)
        meters['teacher_acc5'].update(teacher_acc5.item(), real_batch_size)
        meters['logit_loss'].update(logit_loss.item(), real_batch_size)
        meters['feature_loss'].update(feature_loss.item(), real_batch_size)

        # Note: removed torch.cuda.synchronize() - it was blocking every iteration
        loss_meter.update(loss.item(), real_batch_size)
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        data_tic = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

            if is_main_process() and args.use_wandb:
                acc1_meter, acc5_meter = meters['train_acc1'], meters['train_acc5']
                wandb.log({
                    "train/acc@1": acc1_meter.val,
                    "train/acc@5": acc5_meter.val,
                    "train/loss": loss_meter.val,
                    "train/logit_loss": meters['logit_loss'].val,
                    "train/feature_loss": meters['feature_loss'].val,
                    "train/grad_norm": norm_meter.val,
                    "train/loss_scale": scaler_meter.val,
                    "train/lr": lr,
                }, step=normal_global_idx)

    epoch_time = time.time() - start
    extra_meters_str = f'Train-Summary: [{epoch}/{config.TRAIN.EPOCHS}]\t'
    for k, v in meters.items():
        v.sync()
        extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
    logger.info(extra_meters_str)
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train_one_epoch_online_distill(
    args, config, model, teacher_model, distill_loss,
    data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler
):
    """
    Training with fully online distillation (no saved logits).

    Teacher model performs live forward pass each batch.
    Supports both logits-only and logits+features modes via config.DISTILL.FEATURE_ENABLED.

    Loss = KL(student_logits, teacher_logits) + beta * feature_loss (if enabled)

    This function does NOT require saved teacher logits - everything is computed online.
    """
    from feature_distill import get_student_features

    model.train()
    teacher_model.eval()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    feature_enabled = config.DISTILL.FEATURE_ENABLED

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets

        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_logits, teacher_features = teacher_model(samples)

        # Student forward pass
        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            if feature_enabled:
                student_logits, student_features = get_student_features(model, samples)
            else:
                student_logits = model(samples)
                student_features = None

        # Compute loss (handles both logits-only and logits+features)
        loss, loss_dict = distill_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_features=student_features,
            teacher_features=teacher_features if feature_enabled else None,
        )
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # Backward pass
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        # Compute accuracy
        real_batch_size = len(original_targets)
        with torch.no_grad():
            acc1, acc5 = accuracy(student_logits, original_targets, topk=(1, 5))
            teacher_acc1, teacher_acc5 = accuracy(teacher_logits, original_targets, topk=(1, 5))

        meters['train_acc1'].update(acc1.item(), real_batch_size)
        meters['train_acc5'].update(acc5.item(), real_batch_size)
        meters['teacher_acc1'].update(teacher_acc1.item(), real_batch_size)
        meters['teacher_acc5'].update(teacher_acc5.item(), real_batch_size)
        meters['kl_loss'].update(loss_dict['kl_loss'], real_batch_size)
        if 'feature_loss' in loss_dict:
            meters['feature_loss'].update(loss_dict['feature_loss'], real_batch_size)

        loss_meter.update(loss_dict['total_loss'], real_batch_size)
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

            if is_main_process() and args.use_wandb:
                log_dict = {
                    "train/acc@1": meters['train_acc1'].val,
                    "train/acc@5": meters['train_acc5'].val,
                    "train/loss": loss_meter.val,
                    "train/kl_loss": meters['kl_loss'].val,
                    "train/grad_norm": norm_meter.val,
                    "train/loss_scale": scaler_meter.val,
                    "train/lr": lr,
                }
                if 'feature_loss' in loss_dict:
                    log_dict["train/feature_loss"] = meters['feature_loss'].val
                wandb.log(log_dict, step=normal_global_idx)

    epoch_time = time.time() - start
    extra_meters_str = f'Train-Summary: [{epoch}/{config.TRAIN.EPOCHS}]\t'
    for k, v in meters.items():
        v.sync()
        extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
    logger.info(extra_meters_str)
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(args, config, data_loader, model, num_classes=1000):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        if not args.only_cpu:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            output = model(images)
        if num_classes == 1000:
            output_num_classes = output.size(-1)
            if output_num_classes == 21841:
                output = remap_layer_22kto1k(output)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(
        f' The number of validation samples is {int(acc1_meter.count)}')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    # we follow the throughput measurement of LeViT repo (https://github.com/facebookresearch/LeViT/blob/main/speed_test.py)
    model.eval()

    T0, T1 = 10, 60
    images, _ = next(iter(data_loader))
    batch_size, _, H, W = images.shape
    inputs = torch.randn(batch_size, 3, H, W).cuda(non_blocking=True)

    # trace model to avoid python overhead
    model = torch.jit.trace(model, inputs)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    with torch.amp.autocast('cuda'):
        while time.time() - start < T0:
            model(inputs)
    timing = []
    torch.cuda.synchronize()
    with torch.amp.autocast('cuda'):
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    throughput = batch_size / timing.mean().item()
    logger.info(f"batch_size {batch_size} throughput {throughput}")


if __name__ == '__main__':
    args, config = parse_option()
    config.defrost()
    if config.DISTILL.TEACHER_LOGITS_PATH:
        config.DISTILL.ENABLED = True
    config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    if args.only_cpu:
        ddp_backend = 'gloo'
    else:
        if config.LOCAL_RANK is None:
            config.defrost()
            config.LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
            config.freeze()
        torch.cuda.set_device(config.LOCAL_RANK)
        # Use gloo on Windows (nccl not supported), nccl on Linux
        import platform
        ddp_backend = 'gloo' if platform.system() == 'Windows' else 'nccl'

    # Initialize distributed process group
    # On Windows, use TCPStore without libuv
    if platform.system() == 'Windows':
        import torch.distributed as dist_init
        store = dist_init.TCPStore(
            host_name=os.environ.get('MASTER_ADDR', 'localhost'),
            port=int(os.environ.get('MASTER_PORT', 29500)),
            world_size=world_size,
            is_master=(rank == 0),
            use_libuv=False,  # Disable libuv on Windows
        )
        torch.distributed.init_process_group(
            backend=ddp_backend, store=store, world_size=world_size, rank=rank)
    else:
        torch.distributed.init_process_group(
            backend=ddp_backend, init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Enable TF32 for Ampere+ GPUs (A10, A100, RTX 30xx/40xx) - significant speedup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        config_dict = dict(config)
        config_dict['git'] = get_git_info()
        if args.use_wandb:
            wandb_output_path = config.OUTPUT
            if args.wandb_run_name:
                run_name = args.wandb_run_name
            else:
                run_name = config.MODEL.NAME
            wandb.init(project="TinyViT", config=config_dict,
                       dir=wandb_output_path,name=run_name)

    # print git info
    logger.info('===== git =====')
    logger.info(str(get_git_info()))

    # print config
    logger.info(config.dump())

    main(args, config)
