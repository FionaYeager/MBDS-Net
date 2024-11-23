import sys
import torch
import argparse

from wheel.macosx_libfile import read_data

sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from torch.utils.data import DataLoader
from guided_diffusion.diff_unet import UNet
# from guided_diffusion.second import UNet
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.dataloader import MyTrainData
# from guided_diffusion.bratsloader import MyTrainData
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
import cv2
from guided_diffusion.train_my import TrainLoop

torch.cuda.set_device(0)

def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist(args) #分布式训练
    logger.configure(dir = args.out_dir) #输出文件夹：output
    logger.log("creating data loader...")
    args.in_ch = 4

    train_data = MyTrainData(args.data_dir, train=True, img_scale=1, transform=True)
    datal = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    data = iter(datal)
    # valdata = iter(datal)
    logger.log("creating model and diffusion...")
    # 返回模型和扩散模型
    model1, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model = UNet(4,1)
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    # print(model)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        # valdata = valdata,
        dataloader=datal,
        val_img_ids=None,#test_img_ids,val_img_ids
        # valdataloader=valdatal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_name = 'basedata_new',
        data_dir="/root/Mine/polyp/COVID-19/train",#/root/autodl-tmp/lxf/Mine/polyp/skin/skin
        val_dir="/root/Mine/polyp/COVID-19/val",#/root/autodl-tmp/lxf/Mine/polyp/COVID-19/val
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=1e-8,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=100,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "1",
        maxdice = 0,
        multi_gpu = None, #"0,1,2"
        out_dir='/root/Covid/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
