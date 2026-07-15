# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

import argparse
import logging
import os
import random
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from compressai.datasets import PNGFolder_Trans as PNGFolder
from compressai.losses import RateLoss
from compressai.models.trans_eff import TransJPEGRecompression422 as TransJPEGRecompression
from compressai.optimizers import net_aux_optimizer


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access wrapped module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def build_logger(log_path):
    logger = logging.getLogger("train_lossless_jpeg_trans")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger):
    model.train()
    device = next(model.parameters()).device

    for i, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        y, cb, cr = batch["Y"], batch["Cb"], batch["Cr"]
        batch_size = y.shape[0]
        y, cb, cr = y.to(device), cb.to(device), cr.to(device)

        out_net = model(y, cb, cr)
        out_criterion = criterion(out_net, batch_size, 256)
        (out_criterion["loss"].mean() + 0.0 * out_net["reg"].mean()).backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.mean().backward()
        aux_optimizer.step()

        if i % 10 == 0:
            logger.info(
                "Train epoch %s: [%s/%s (%s%%)]\tLoss: %.3f |\tRec loss: %.3f |\tBpp loss: %.2f |\tAux loss: %.2f",
                epoch,
                i * batch_size,
                len(train_dataloader.dataset),
                100.0 * (i * batch_size) / len(train_dataloader.dataset),
                out_criterion["loss"].mean().item(),
                out_criterion["rec_loss"].mean().item(),
                out_criterion["bpp_loss"].mean().item(),
                aux_loss.mean().item(),
            )


@torch.no_grad()
def evaluate(epoch, dataloader, model, criterion, logger, split_name):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    rec_loss = AverageMeter()
    aux_loss = AverageMeter()

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        y, cb, cr = batch["Y"], batch["Cb"], batch["Cr"]
        batch_size = y.shape[0]
        y, cb, cr = y.to(device), cb.to(device), cr.to(device)
        out_net = model(y, cb, cr)
        out_criterion = criterion(out_net, batch_size, 256)

        aux_loss.update(model.aux_loss().mean())
        bpp_loss.update(out_criterion["bpp_loss"].mean())
        loss.update(out_criterion["loss"].mean())
        rec_loss.update(out_criterion["rec_loss"].mean())

    logger.info(
        "%s epoch %s: Average losses:\tLoss: %.3f |\tRec loss: %.3f |\tBpp loss: %.2f |\tAux loss: %.2f",
        split_name,
        epoch,
        loss.avg,
        rec_loss.avg,
        bpp_loss.avg,
        aux_loss.avg,
    )
    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace(".pth.tar", "_best_loss.pth.tar"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train the pathology JPEG recompression model.")
    parser.add_argument("-m", "--model", default="eff_jpeg", help="Model tag used for naming outputs.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the training dataset root.")
    parser.add_argument("-e", "--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Main learning rate.")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("-q", "--quality", type=int, default=75, help="JPEG quality used during training data generation.")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=1e-2, help="Rate-distortion tradeoff parameter.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, default=64, help="Validation/test batch size.")
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3, help="Auxiliary loss learning rate.")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Input patch size.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--rle", action="store_true", help="Enable RLE branch.")
    parser.add_argument("--gaussian", action="store_true", help="Use Gaussian entropy model instead of Laplace.")
    parser.add_argument("--save", action="store_true", default=True, help="Save checkpoints.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--clip-max-norm", default=1.0, type=float, help="Gradient clipping max norm.")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint for resuming training.")
    parser.add_argument("--method", type=str, default="pathology_jpeg_trans", help="Experiment tag.")
    parser.add_argument("--net", default="B", type=str, help="Backbone scale: B, L, or Huge.")
    parser.add_argument("--chunk", nargs="+", default=("scales", "means"), help="Chunk parameters stored for bookkeeping.")
    parser.add_argument("--output-dir", type=str, default="compress_output", help="Directory used to save logs and checkpoints.")
    parser.add_argument("--experiment-name", type=str, default="", help="Optional custom experiment name.")
    parser.add_argument("--train-prefetch-factor", type=int, default=4, help="Prefetch factor for the training dataloader.")
    return parser.parse_args(argv)


def make_experiment_dir(args):
    if args.experiment_name:
        run_name = args.experiment_name
    else:
        run_name = f"{args.model}_{args.method}_q{args.quality}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    run_dir = make_experiment_dir(args)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    chunk_tag = "-".join(args.chunk)
    log_path = os.path.join(run_dir, f"train_{timestamp}_{chunk_tag}.log")
    logger = build_logger(log_path)
    logger.info("Arguments: %s", vars(args))

    train_dataset = PNGFolder(args.dataset, mode="train", quality=args.quality)
    test_dataset = PNGFolder(args.dataset, mode="test", quality=args.quality)
    val_dataset = PNGFolder(args.dataset, mode="val", quality=args.quality)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        prefetch_factor=args.train_prefetch_factor if args.num_workers > 0 else None,
        pin_memory=(device == "cuda"),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = TransJPEGRecompression(chunk=("scales", "means"), net=args.net, rle=args.rle, gaussian=args.gaussian)
    logger.info(net)
    net = net.to(device)

    logger.info("Using entropy model: %s", "Gaussian" if args.gaussian else "Laplace")
    if args.rle:
        logger.info("RLE branch enabled")
    if args.cuda and torch.cuda.device_count() > 1:
        logger.info("Using %s GPUs", torch.cuda.device_count())
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateLoss()

    last_epoch = 0
    if args.checkpoint:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    checkpoint_path = os.path.join(run_dir, "checkpoint.pth.tar")

    for epoch in range(last_epoch, args.epochs):
        logger.info("Epoch %s learning rate: %s", epoch, optimizer.param_groups[0]["lr"])
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger,
        )

        loss = evaluate(epoch, val_dataloader, net, criterion, logger, split_name="Val")
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename=checkpoint_path,
            )

    logger.info("Best validation loss: %s", best_loss)

    best_checkpoint_path = checkpoint_path.replace(".pth.tar", "_best_loss.pth.tar")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        evaluate(args.epochs - 1, test_dataloader, net, criterion, logger, split_name="Test")
    else:
        logger.warning("Best checkpoint not found at %s; skipping final test evaluation.", best_checkpoint_path)


if __name__ == "__main__":
    main(sys.argv[1:])
