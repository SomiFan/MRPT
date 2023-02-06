"""
finetune.py 2022/6/27 14:07
Written by Wensheng Fan
"""
import argparse
from cfg import get_config_ft

import numpy as np
import random
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from tensorboardX import SummaryWriter
from models import build_model
from data import build_concat_loader
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from utils import auto_resume_helper_ft, load_checkpoint_ft, save_checkpoint_ft, create_logger
from train_functions import throughput, train_one_epoch_ft, validateft, train_ft_lwfs
from test_functions import reduced_res_test, full_res_test
from loss_funcs import SiSLoss, EdgeLoss, MMLwFLoss, QNRLoss


def parse_option():
    parser = argparse.ArgumentParser('HRPFormer finetune script', add_help=False)
    parser.add_argument(
        "--cfg",
        type=str,
        metavar="FILE",
        default=r"./configs/pnn/pnn_ft.yaml",
        help="path to config file, model name and type are in the yaml cfg file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, default=2, help="batch size for single GPU")
    parser.add_argument("--tag", type=str, default="qb", help="tag of experiment")
    parser.add_argument("--pret-ckpt", type=str, default="best_ergas", choices=["best_ergas", "min_loss", "latest", "best_sam"],
                        help="the model to be fine-tuned")
    parser.add_argument("--ckpt-choice", type=str, default="best_qnr", choices=["min_loss", "latest", "best_qnr"],
                        help="the model to be tested")
    parser.add_argument("--test", action="store_true", help="Perform test only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )
    # set training dataset
    parser.add_argument(
        "--data-set", default="qb", choices=["gf2", "qb", "wv3", "gf2l", "spot", "qbsx"], type=str
    )
    parser.add_argument(
        "--tsp", default="", type=str, help="Training Set Path"
    )
    parser.add_argument('--base-lr', type=float, default=0.0005, help='basic learning rate')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='threads for data loading, too many would cause error, for ur own pc suggest 1')
    parser.add_argument("--verbose", action="store_false", help="output verbose for analysis")
    parser.add_argument('--gpu-set', type=int, default=1, help='if >1, should modify the code otherwise wont be valid')

    args, unparsed = parser.parse_known_args()
    config = get_config_ft(args)
    return args, config


def main(config):
    (
        train_loader,
        val_loader,
    ) = build_concat_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model_o = build_model(config)
    model = build_model(config)

    optimizer = build_optimizer(config, model)

    model_o = torch.nn.DataParallel(model_o, device_ids=gpus).cuda()
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    logger.info(str(model))

    # 模型参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    # flops
    model_o_without_ddp = model_o.module
    model_without_ddp = model.module
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # Define Losses
    criterion = {}

    # criterion['qnr'] = QNRLoss().cuda()
    criterion['sis'] = SiSLoss(ns=config.TRAIN.SIS_NS).cuda()
    criterion['grad'] = EdgeLoss().cuda()
    if config.TRAIN.LWF:
        criterion['lwf_real'] = MMLwFLoss().cuda() if config.TRAIN.MMLWF else nn.L1Loss().cuda()
        if config.TRAIN.LWFS:
            criterion['lwf_syn'] = MMLwFLoss().cuda() if config.TRAIN.MMLWF else nn.L1Loss().cuda()

    config.defrost()
    config.PRINT_FREQ = len(train_loader) - 1
    config.freeze()

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper_ft(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    # 因为ckpt load到cpu上，所以这里把model_without_ddp传进去
    min_loss, best_qnr = load_checkpoint_ft(config, model_without_ddp, model_o_without_ddp, optimizer, lr_scheduler, logger)

    if config.THROUGHPUT_MODE:
        throughput(config, val_loader, model, logger)
        return

    if config.TEST_MODE:
        logger.info("Start reduced resolution test")
        reduced_res_test(config, model, logger)
        logger.info("Start full resolution test")
        full_res_test(config, model, logger)
        return

    writer = SummaryWriter(config.OUTPUT)
    writer_dict = {
        'writer': writer,
        'train_global_steps': config.TRAIN.START_EPOCH * len(train_loader),
    }

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # train_loader.sampler.set_epoch(epoch)
        is_best_qnr = False
        is_min_loss = False

        # for param in model_o.parameters():
        #     param.requires_grad = False

        if config.TRAIN.LWFS:
            train_ft_lwfs(
                config,
                model,
                model_o,
                criterion,
                train_loader,
                optimizer,
                epoch,
                lr_scheduler,
                logger,
                writer_dict,
            )
        else:
            train_one_epoch_ft(
                config,
                model,
                model_o,
                criterion,
                train_loader,
                optimizer,
                epoch,
                lr_scheduler,
                logger,
                writer_dict,
            )
        if epoch % config.VAL_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            loss, qnr = validateft(config, val_loader, model, model_o, criterion, logger, epoch, writer_dict)
            if qnr > best_qnr:
                best_qnr = qnr
                is_best_qnr = True
            else:
                is_best_qnr = False
            if loss < min_loss:
                min_loss = loss
                is_min_loss = True
            else:
                is_min_loss = False

        save_checkpoint_ft(
            config,
            epoch,
            model_without_ddp,
            min_loss,
            is_min_loss,
            best_qnr,
            is_best_qnr,
            optimizer,
            lr_scheduler,
            logger
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    logger.info(f"Best QNR: {best_qnr}")


if __name__ == "__main__":
    _, config = parse_option()

    if torch.cuda.is_available():
        # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        cudnn.benchmark = True
    # gpus = [gpu for gpu in range(config.GPUSET)]
    # torch.cuda.set_device(gpus[0])
    gpus = [0]
    torch.cuda.set_device(0)

    # 固定随机数种子，只要输入一样，每次计算的结果也一样
    if config.SEED is not None:
        torch.manual_seed(config.RANDOM_SEED)
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        random.seed(config.RANDOM_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}_finetune"
    )
    path = os.path.join(config.OUTPUT, "config_finetune.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
