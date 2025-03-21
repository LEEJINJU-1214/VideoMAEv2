# --------------------------------------------------------
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import io
import os
import json
import math
import time
import random
import datetime
import subprocess
from typing import Tuple
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import inf
from timm.utils import get_state_dict
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data._utils.collate import default_collate
from datetime import timedelta

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device="cuda"
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def min(self):
        return min(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            min=self.min,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr
            )
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f} ({min:.4f} -- {max:.4f})")
        data_time = SmoothedValue(fmt="{avg:.4f} ({min:.4f} -- {max:.4f})")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head="scalar", step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(
                head + "/" + k, v, self.step if step is None else step
            )

    def flush(self):
        self.writer.flush()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = int(os.environ["SLURM_LOCALID"])
        args.world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["RANK"] = str(args.rank)
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["WORLD_SIZE"] = str(args.world_size)

        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "gloo"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(seconds=1800) 
    )
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    assert torch.distributed.is_initialized()
    setup_for_distributed(args.rank == 0)


def load_state_dict(
    model, state_dict, prefix="", ignore_missing="relative_position_index"
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = (
            {} if metadata is None else metadata.get(prefix[:-1], {})
        )
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(
            p.grad.detach().abs().max().to(device) for p in parameters
        )
    else:
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), norm_type).to(device)
                    for p in parameters
                ]
            ),
            norm_type,
        )
    return total_norm


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
    warmup_steps=-1,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters
        )

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(
    args,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    model_ema=None,
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            if model_ema is not None:
                to_save["model_ema"] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        if model_ema is not None:
            client_state["model_ema"] = get_state_dict(model_ema)
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )


def auto_load_model(
    args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None
):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob

            all_checkpoints = glob.glob(
                os.path.join(output_dir, "checkpoint-*.pth")
            )
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(
                    output_dir, "checkpoint-%d.pth" % latest_ckpt
                )
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            print("Resume checkpoint %s" % args.resume)
            if "optimizer" in checkpoint and "epoch" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"] + 1
                if hasattr(args, "model_ema") and args.model_ema:
                    _load_checkpoint_for_ema(
                        model_ema, checkpoint["model_ema"]
                    )
                if "scaler" in checkpoint:
                    loss_scaler.load_state_dict(checkpoint["scaler"])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob

            all_checkpoints = glob.glob(
                os.path.join(output_dir, "checkpoint-*")
            )
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(
                    output_dir, "checkpoint-%d" % latest_ckpt
                )
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(
                    args.output_dir, tag="checkpoint-%d" % latest_ckpt
                )
                if "epoch" in client_states:
                    args.start_epoch = client_states["epoch"] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(
                            model_ema, client_states["model_ema"]
                        )


def create_ds_config(args):
    args.deepspeed_config = os.path.join(
        args.output_dir, "deepspeed_config.json"
    )
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size
            * args.update_freq
            * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "gradient_clipping": 0.0
            if args.clip_grad is None
            else args.clip_grad,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128,
            },
            # 'allreduce_bucket_size':   args.allreduce_bucket_size,
            # "train_micro_batch_size_per_gpu": args.batch_size,
            # "gradient_accumulation_steps": args.update_freq,
        }

        writer.write(json.dumps(ds_config, indent=2))


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data


def multiple_pretrain_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    process_data, encoder_mask, decoder_mask = zip(*batch)

    process_data = [item for sublist in process_data for item in sublist]
    encoder_mask = [item for sublist in encoder_mask for item in sublist]
    decoder_mask = [item for sublist in decoder_mask for item in sublist]
    process_data, encoder_mask, decoder_mask = (
        default_collate(process_data),
        default_collate(encoder_mask),
        default_collate(decoder_mask),
    )
    if fold:
        return [process_data], encoder_mask, decoder_mask
    else:
        return process_data, encoder_mask, decoder_mask


def to_normalized_float_tensor(vid: torch.Tensor) -> torch.Tensor:
    """영상 텐서의 채널 순서를 변경하고 [0, 255] 범위를 [0, 1]로 정규화한다.

    (T, H, W, C) 순서의 텐서를 (C, T, H, W)로 permute한 뒤,
    각 픽셀 값을 float32 형식으로 255.0으로 나눈다.

    Args:
        vid (torch.Tensor): 4차원 텐서 (T, H, W, C).

    Returns:
        torch.Tensor: (C, T, H, W) 형태로 permute되고 [0, 1]로 정규화된 텐서.
    """
    return vid.permute(3, 0, 1, 2).float() / 255.0


def resize(
    vid: torch.Tensor, size, interpolation: str = "bilinear"
) -> torch.Tensor:
    """영상 텐서의 공간적 크기를 지정된 size로 조정한다.

    (C, T, H, W) 형태의 텐서에 대해, bilinear 등 보간을 사용해 크기를 변경한다.
    size가 int일 경우, 더 작은 쪽을 size에 맞추고 비율에 따라 다른 쪽을 조정한다.
    size가 (H, W) 튜플이면 해당 크기로 변경한다.

    Args:
        vid (torch.Tensor): 4차원 텐서 (C, T, H, W).
        size (int or tuple): 목표 크기. 정수(int) 또는 (H, W) 형태의 튜플.
        interpolation (str, optional): 보간 모드. 기본값은 "bilinear".

    Returns:
        torch.Tensor: 크기가 조정된 4차원 텐서 (C, T, H', W').
    """
    if isinstance(size, int):
        # 작은 쪽을 size로 맞추기 위한 scale_factor 계산
        scale = float(size) / min(vid.shape[-2:])
        return nn.functional.interpolate(
            vid, scale_factor=scale, mode=interpolation, align_corners=False
        )
    else:
        return nn.functional.interpolate(
            vid, size=size, mode=interpolation, align_corners=False
        )


class ToFloatTensorInZeroOne:
    """OpenCV 포맷의 영상 텐서를 0~1 범위 float 텐서로 변환하는 변환 클래스."""

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        """영상을 [0, 1] 범위로 정규화된 float 텐서로 변환한다.

        Args:
            vid (torch.Tensor): (T, H, W, C) 형태의 텐서.

        Returns:
            torch.Tensor: (C, T, H, W) 형태의 [0, 1] float 텐서.
        """
        return to_normalized_float_tensor(vid)


class Resize:
    """4차원 영상 텐서를 지정한 크기로 공간적 리사이즈하는 변환 클래스."""

    def __init__(self, size):
        """Resize 클래스 생성자.

        Args:
            size (int or tuple): 목표 크기. 정수(int) 또는 (H, W) 튜플.
        """
        self.size = size

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        """영상을 주어진 크기로 리사이즈한다.

        Args:
            vid (torch.Tensor): (C, T, H, W) 형태의 텐서.

        Returns:
            torch.Tensor: 크기가 조정된 텐서.
        """
        return resize(vid, self.size)


def get_transform(size: Tuple[int, int]) -> transforms.Compose:
    """영상 텐서에 대한 전처리(정규화 + 리사이즈) 파이프라인을 구성한다.

    Args:
        size (Tuple[int, int]): (H, W) 형태의 목표 크기.

    Returns:
        transforms.Compose: ToFloatTensorInZeroOne -> Resize 순으로 적용되는 Transform.
    """
    return transforms.Compose(
        [
            ToFloatTensorInZeroOne(),
            Resize(size),
        ]
    )
