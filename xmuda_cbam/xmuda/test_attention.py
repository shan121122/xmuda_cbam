#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch

from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build_cbam import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('ckpt2d', type=str, help='path to checkpoint file of the 2D model')
    parser.add_argument('ckpt3d', type=str, help='path to checkpoint file of the 3D model')
    parser.add_argument('--pselab', action='store_true', help='generate pseudo-labels')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def test(cfg, args, output_dir=''):
    logger = logging.getLogger('xmuda.test')

    # build 2d model
    model_2d = build_model_2d(cfg)[0]

    # build 3d model
    model_3d = build_model_3d(cfg)[0]

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build checkpointer
    checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=logger)
    checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=logger)
    
    # Load checkpoints with strict=False to handle architecture mismatches
    logger.info('Loading 2D model checkpoint from {}'.format(args.ckpt2d))
    checkpoint_data_2d = checkpointer_2d.load(args.ckpt2d.replace('@', output_dir), 
                                            resume=False, 
                                            strict=False)
    
    if checkpoint_data_2d is None:
        raise RuntimeError('Failed to load 2D model checkpoint')
    
    logger.info('Loading 3D model checkpoint from {}'.format(args.ckpt3d))
    checkpoint_data_3d = checkpointer_3d.load(args.ckpt3d.replace('@', output_dir), 
                                            resume=False, 
                                            strict=False)
    if checkpoint_data_3d is None:
        raise RuntimeError('Failed to load 3D model checkpoint')

    # Log model architectures
    logger.info('2D Model architecture:\n{}'.format(model_2d))
    logger.info('3D Model architecture:\n{}'.format(model_3d))

    # build dataset
    test_dataloader = build_dataloader(cfg, mode='test', domain='target')

    pselab_path = None
    if args.pselab:
        pselab_dir = osp.join(output_dir, 'pselab_data')
        os.makedirs(pselab_dir, exist_ok=True)
        assert len(cfg.DATASET_TARGET.TEST) == 1
        pselab_path = osp.join(pselab_dir, cfg.DATASET_TARGET.TEST[0] + '.npy')
        logger.info('Will save pseudo labels to {}'.format(pselab_path))

    # Test
    set_random_seed(cfg.RNG_SEED)
    test_metric_logger = MetricLogger(delimiter='  ')
    model_2d.eval()
    model_3d.eval()

    validate(cfg, model_2d, model_3d, test_dataloader, test_metric_logger, pselab_path=pselab_path)

def main():
    args = parse_args()

    # load the configuration
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if not osp.isdir(output_dir):
            warnings.warn('Making new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('Available GPUs: {:d}'.format(torch.cuda.device_count()))
    logger.info('Command line args: {}'.format(args))
    logger.info('Loaded config file: {}'.format(args.config_file))
    logger.info('Full config:\n{}'.format(cfg))

    # Verify model configuration
    logger.info('MODEL_2D config:')
    logger.info('- DUAL_HEAD: {}'.format(cfg.MODEL_2D.DUAL_HEAD))
    logger.info('- NUM_CLASSES: {}'.format(cfg.MODEL_2D.NUM_CLASSES))
    logger.info('- FEATURE_DIM: {}'.format(cfg.MODEL_2D.FEATURE_DIM if hasattr(cfg.MODEL_2D, 'FEATURE_DIM') else 'N/A'))

    logger.info('MODEL_3D config:')
    logger.info('- DUAL_HEAD: {}'.format(cfg.MODEL_3D.DUAL_HEAD))
    logger.info('- NUM_CLASSES: {}'.format(cfg.MODEL_3D.NUM_CLASSES))

    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD, "2D and 3D models must have same DUAL_HEAD setting"
    
    test(cfg, args, output_dir)


if __name__ == '__main__':
    main()