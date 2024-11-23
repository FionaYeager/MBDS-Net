from typing import List

import ml_collections
import torch
from pathlib import Path
import os


def create_argparser():
    defaults = dict(
        data_name = 'basedata_new',
        data_dir="/root/baseline/input/basedata_new/train/",
        val_dir="/root/baseline/input/basedata_new/test/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=1e-8,
        lr_anneal_steps=100000,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=241,
        save_interval=100,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        maxdice = 0,
        multi_gpu = None, #"0,1,2"
        out_dir='outputs/unet++/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
