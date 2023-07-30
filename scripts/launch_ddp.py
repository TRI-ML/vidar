# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os

import fire
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vidar.core.trainer import Trainer
from vidar.utils.config import read_config
from vidar.core.wrapper import Wrapper


def train(cfg, **kwargs):

    os.environ['DIST_MODE'] = 'ddp'

    cfg = read_config(cfg, **kwargs)

    mp.spawn(main_worker,
             nprocs=torch.cuda.device_count(),
             args=(cfg,), join=True)


def main_worker(gpu, cfg):

    torch.cuda.set_device(gpu)
    world_size = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    os.environ['RANK'] = str(gpu)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['DIST_MODE'] = 'ddp'

    dist.init_process_group(backend='nccl', world_size=world_size, rank=gpu)

    wrapper = Wrapper(cfg, verbose=True)
    trainer = Trainer(cfg)
    trainer.learn(wrapper)

    dist.destroy_process_group()


if __name__ == '__main__':
    fire.Fire(train)
