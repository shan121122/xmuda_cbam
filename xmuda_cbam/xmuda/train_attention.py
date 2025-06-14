#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from xmuda.common.solver.build import build_optimizer, build_scheduler
from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build_cbam import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate
from xmuda.models.losses import entropy_loss

from xmuda.models.cbam import CBAM
from xmuda.models.feature_fusion import FeatureFusionModule

def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger


def train(cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('xmuda.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    cbam_2d = CBAM(cfg.MODEL_2D.FEATURE_DIM)
    cbam_3d = CBAM(cfg.MODEL_3D.FEATURE_DIM)
    fusion_module = FeatureFusionModule(cfg.MODEL_2D.FEATURE_DIM, cfg.MODEL_3D.FEATURE_DIM, cfg.CBAM.FUSION_OUT_CHANNELS)
    fusion_classifier = nn.Linear(cfg.CBAM.FUSION_OUT_CHANNELS, cfg.MODEL_2D.NUM_CLASSES).cuda()
    
    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()
    cbam_2d = cbam_2d.cuda()
    cbam_3d = cbam_3d.cuda()
    fusion_module = fusion_module.cuda()

    # build optimizer
    optimizer_2d = build_optimizer(cfg, model_2d)
    optimizer_3d = build_optimizer(cfg, model_3d)
    optimizer_cbam_2d = build_optimizer(cfg, cbam_2d)
    optimizer_cbam_3d = build_optimizer(cfg, cbam_3d)
    optimizer_fusion = build_optimizer(cfg, fusion_module)

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)
    scheduler_cbam_2d = build_scheduler(cfg, optimizer_cbam_2d)
    scheduler_cbam_3d = build_scheduler(cfg, optimizer_cbam_3d)
    scheduler_fusion = build_scheduler(cfg, optimizer_fusion)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_cbam_2d = CheckpointerV2(cbam_2d,
                                        optimizer=optimizer_cbam_2d,
                                        scheduler=scheduler_cbam_2d,
                                        save_dir=output_dir,
                                        logger=logger,
                                        postfix='_cbam_2d',
                                        max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpointer_cbam_3d = CheckpointerV2(cbam_3d,
                                        optimizer=optimizer_cbam_3d,
                                        scheduler=scheduler_cbam_3d,
                                        save_dir=output_dir,
                                        logger=logger,
                                        postfix='_cbam_3d',
                                        max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpointer_fusion = CheckpointerV2(fusion_module,
                                       optimizer=optimizer_fusion,
                                       scheduler=scheduler_fusion,
                                       save_dir=output_dir,
                                       logger=logger,
                                       postfix='_fusion',
                                       max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None),
        'fusion': None
    }
    best_metric_iter = {'2d': -1, '3d': -1, 'fusion': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        cbam_2d.train()
        cbam_3d.train()
        fusion_module.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        cbam_2d.eval()
        cbam_3d.eval()
        fusion_module.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)
    for iteration in range(start_iteration, max_iteration):
        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        data_time = time.time() - end
        # copy data from cpu to gpu
        if 'SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.DATASET_TARGET.TYPE:
            # source
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
            # target
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
                data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()
        optimizer_cbam_2d.zero_grad()
        optimizer_cbam_3d.zero_grad()
        optimizer_fusion.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_src)
        preds_3d = model_3d(data_batch_src)

        # CBAM attention
        feat_2d = preds_2d['feat']  # [B, C, H, W]
        feat_3d = preds_3d['feat']  # [N, C]
        feat_2d_cbam = cbam_2d(feat_2d)  # 输入应为特征张量
        feat_3d_cbam = cbam_3d(feat_3d)
        
        # 融合
        proj_indices = data_batch_src['img_indices']
        fused_feat = fusion_module(feat_2d_cbam, feat_3d_cbam, proj_indices)
        assert isinstance(fused_feat, torch.Tensor), "Fused features must be Tensor"
        fused_logit = fusion_classifier(fused_feat)

        # segmentation loss: cross entropy
        seg_loss_src_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        seg_loss_src_fusion = F.cross_entropy(fused_logit, data_batch_src['seg_label'], weight=class_weights)
        
        train_metric_logger.update(seg_loss_src_2d=seg_loss_src_2d, seg_loss_src_3d=seg_loss_src_3d,seg_loss_src_fusion=seg_loss_src_fusion)
        
        loss_2d = seg_loss_src_2d
        loss_3d = seg_loss_src_3d
        loss_fusion = seg_loss_src_fusion

        # 加权融合损失（可选）
        if cfg.CBAM.LOSS_WEIGHT > 0:
            fused_loss = F.cross_entropy(fused_logit, data_batch_src['seg_label'], weight=class_weights)
            train_metric_logger.update(fused_loss_src=fused_loss)
            loss_fusion = cfg.CBAM.LOSS_WEIGHT * fused_loss
        if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                                       xm_loss_src_3d=xm_loss_src_3d)
            loss_2d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
            loss_3d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d
        
        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, data_batch_src)
            train_metric_3d.update_dict(preds_3d, data_batch_src)

        # Backward pass for source data
        # loss_2d.backward()
        # loss_3d.backward()
        # loss_fusion.backward()
        total_loss = loss_2d + loss_3d + loss_fusion
        total_loss.backward()

        optimizer_2d.step()
        optimizer_3d.step()
        optimizer_cbam_2d.step()
        optimizer_cbam_3d.step()
        optimizer_fusion.step()

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #
        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()
        optimizer_cbam_2d.zero_grad()
        optimizer_cbam_3d.zero_grad()
        optimizer_fusion.zero_grad()
        
        preds_2d = model_2d(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)
        feat_2d_cbam = cbam_2d(preds_2d['feat'])
        feat_3d_cbam = cbam_3d(preds_3d['feat'])

        proj_indices_trg = data_batch_trg['img_indices']
        fused_feat = fusion_module(feat_2d_cbam, feat_3d_cbam, proj_indices_trg)
        fused_logit = fusion_classifier(fused_feat)
        
        loss_2d = 0
        loss_3d = 0
        loss_fusion = 0

        if cfg.CBAM.LOSS_WEIGHT > 0:
            fused_loss = F.cross_entropy(fused_logit, data_batch_trg['seg_label'], weight=class_weights)
            train_metric_logger.update(fused_loss_trg=fused_loss)
            loss_fusion = cfg.CBAM.LOSS_WEIGHT * fused_loss
        if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d += cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_2d
            loss_3d += cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_3d
        if cfg.TRAIN.XMUDA.lambda_pl > 0:
            # uni-modal self-training loss with pseudo labels
            pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_trg['pseudo_label_2d'])
            pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_trg['pseudo_label_3d'])
            train_metric_logger.update(pl_loss_trg_2d=pl_loss_trg_2d,
                                       pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d += cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d
            loss_3d += cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d
        if cfg.TRAIN.XMUDA.lambda_minent > 0:
            # MinEnt
            minent_loss_trg_2d = entropy_loss(F.softmax(preds_2d['seg_logit'], dim=1))
            minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d['seg_logit'], dim=1))
            train_metric_logger.update(minent_loss_trg_2d=minent_loss_trg_2d,
                                       minent_loss_trg_3d=minent_loss_trg_3d)
            loss_2d += cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_2d
            loss_3d += cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_3d

        # loss_2d.backward()
        # loss_3d.backward()
        # loss_fusion.backward()
        total_loss_trg = loss_2d + loss_3d + loss_fusion
        total_loss_trg.backward()

        optimizer_2d.step()
        optimizer_3d.step()
        optimizer_cbam_2d.step()
        optimizer_cbam_3d.step()
        optimizer_fusion.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)
            checkpointer_cbam_2d.save('cbam_2d_{:06d}'.format(cur_iter), iteration=cur_iter)
            checkpointer_cbam_3d.save('cbam_3d_{:06d}'.format(cur_iter), iteration=cur_iter)
            checkpointer_fusion.save('fusion_{:06d}'.format(cur_iter), iteration=cur_iter)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d', '3d', 'fusion']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        scheduler_cbam_2d.step()
        scheduler_cbam_3d.step()
        scheduler_fusion.step()

        end = time.time()

    for modality in ['2d', '3d', 'fusion']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
           cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg, output_dir, run_name)


if __name__ == '__main__':
    main()
