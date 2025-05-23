import copy
import functools
import os
from guided_diffusion.utils import dice_score, iou_score2, haussdorf
from guided_diffusion.dataloader import MyTrainData
# from guided_diffusion.bratsloader import MyTrainData
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from guided_diffusion.gaussian_diffusion import DiceLoss

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        val_img_ids,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.dataloader=dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.val_ids = val_img_ids
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size# * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()

            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = False
            self.ddp_model = self.model
        else:
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        # dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        self.maxdice = 0
        while (
            not self.lr_anneal_steps
            and self.step + self.resume_step < 100000
        ):
            try:
                    batch, cond = next(data_iter)
                    # print(batch)
            except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch, cond = next(data_iter)

            self.run_step(batch, cond)
            i += 1
            with th.no_grad():
                if (self.step) % 100 == 0:
                    logger.dumpkvs()
                if self.step + self.resume_step > 20000 and (self.step + self.resume_step) % 1000 == 0:  # self.step + self.resume_step > 44000,
                    # print("ooo")
                    ds = MyTrainData("/root/Mine/polyp/COVID-19/val", train=False,
                                     img_scale=1, transform=False)  #, img_ids=self.val_ids
                    datal = th.utils.data.DataLoader(
                        ds,
                        batch_size=16,
                        shuffle=False)
                    data = iter(datal)
                    dice = 0
                    iou = 0
                    hauss = 0
                    loss = 0
                    for val_step in range(25):  # 242(CLITI 31),201/67(cell 26/9/5),447(56),411(brain 52),skin259/17,brain137/9 refuge
                        # print("val_step", val_step)
                        try:
                            b, ma, path = next(data)
                            # print("batch:",batch)
                        except StopIteration:
                            valdata_iter = iter(datal)
                            b, ma, path = next(valdata_iter)
                        # b, m, path = next(valdata_iter)  # should return an image from the dataloader "data"
                        c = th.randn_like(b[:, :1, ...])
                        img = th.cat((b, c), dim=1)
                        m, n, h, lo = self.valforward(img, ma)
                        dice = dice + m
                        iou = iou + n
                        hauss = hauss + h
                        loss = loss + lo
                    val_step = val_step + 1
                    dice = dice / val_step
                    iou = iou / val_step
                    hauss = hauss / val_step
                    loss = loss / val_step
                    if dice > self.maxdice:
                        self.maxdice = dice
                        self.save()
                    else:
                        self.save()
                    print("max_dice", self.maxdice)
                    logger.log(f"step:{self.step + self.resume_step}...val_dice:{dice}")
                    logger.logkv("val_dice", dice)
                    logger.logkv("val_iou", iou)
                    logger.logkv("val_haus", hauss)
                    logger.logkv("max_dice", self.maxdice)
                    logger.logkv("val_loss", loss)
                    logger.dumpkvs()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

            self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def valforward(self, img, m):
        self.ddp_model.eval()
        i = 0
        # dice = 0
        haus = 0
        # loss = 0
        logger.log("sampling...")
        lloss = DiceLoss()
        for i in range(1):  # this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            sample_fn = (
                self.diffusion.p_sample_loop_known #if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, pred_start = sample_fn(
                self.ddp_model,
                (img.shape[0], 4, 256, 256), img,
                step=1000,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )
            pred_start = th.where(pred_start > 0.5, 1, 0).squeeze(1)
        dice = dice_score(pred_start.cpu().numpy(), m.squeeze(1).cpu().numpy())
        iou = iou_score2(pred_start.cpu().numpy(), m.squeeze(1).cpu().numpy())
        loss = lloss(pred_start.cpu(), m.cpu())
        for output, mask_gt in zip(pred_start, m.squeeze(1)):
            i = i + 1
            mask_pred = (output > 0.5).float()  # mask_pred.size()(1,256,256),mask_gt.size()(1,256,256)
            haus = haus + haussdorf(mask_pred, mask_gt)
        return dice, iou, haus/i, loss
        # logger.log(f"dice: {dice}")

    def run_step(self, batch, cond):
        batch=th.cat((batch, cond), dim=1)

        cond={}
        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond):
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            # print("t:",t.shape)#8
            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            losses = losses1[0]
            sample = losses1[1]

            # loss = (losses["loss"] * weights).mean()
            loss = losses["loss"]
            # print("loss:",weights)

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            # self.mp_trainer.zero_grad()
            self.opt.zero_grad()
            self.mp_trainer.backward(loss)
            self.opt.step()
            return sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # if dist.get_rank() == 0:
            #     logger.log(f"saving model {rate}...")
            #     if not rate:
            #         filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
            #         # filename = f"savedmodel.pt"
            #     else:
            #         # filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
            #         filename = f"emasavedmodel.pt"
            #     with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
            #         th.save(state_dict, f)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"savedmodel{(self.step + self.resume_step):06d}.pt"
                # filename = f"savedmodel.pt"
            else:
                # filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                filename = f"emasavedmodel.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    return logger.get_dir()


def find_resume_checkpoint():
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
