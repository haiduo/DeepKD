import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
import time

# cudnn.benchmark = True

def setup_seed(seed):
    """
    Set random seed for reproducibility in PyTorch, NumPy, and Python's random module.
    
    Args:
        seed (int): The seed value to be set.
    """
    # Set random seed for PyTorch (CPU)
    torch.manual_seed(seed)
    # Set random seed for NumPy
    np.random.seed(seed)
    # Set random seed for Python's random module
    random.seed(seed)
    # Set hash seed for Python (deterministic hash-based operations)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set seed for GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
    
    print(f'Random seed set to: {seed}')

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
from timm.models import create_model
def main(cfg, resume, opts,cfg_path):
    start_time = time.time()  
    
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)

    
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    if cfg.DISTILLER.TYPE == 'MLKD' or cfg.DISTILLER.TYPE=='CRLD':
        train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)
    else:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)
    
    end_time = time.time()  
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\nTotal training time: {hours} hours {minutes} minutes {seconds} seconds")
    # Write training time to log file
    with open(os.path.join(cfg.LOG.PREFIX,experiment_name, "worklog.txt"), "a") as writer:
        writer.write(f"\nTotal training time: {hours} hours {minutes} minutes {seconds} seconds\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="configs/cifar100/dkd/resnet110_resnet20.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logit-stand", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--seed', default=42, type=int, metavar='N', help='random seed for reproducibility')
    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    if args.logit_stand and cfg.DISTILLER.TYPE in ['KD','DKD','MLKD']:
        cfg.EXPERIMENT.LOGIT_STAND = True
        if cfg.DISTILLER.TYPE == 'KD':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp
        elif cfg.DISTILLER.TYPE == 'DKD':
            cfg.DKD.ALPHA = cfg.DKD.ALPHA * args.kd_weight
            cfg.DKD.BETA = cfg.DKD.BETA * args.kd_weight
            cfg.DKD.T = args.base_temp
        elif cfg.DISTILLER.TYPE == 'MLKD':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp
    cfg.freeze()
    setup_seed(args.seed)
    
    main(cfg, args.resume, args.opts,args.cfg)
