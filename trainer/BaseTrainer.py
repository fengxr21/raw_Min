import sys
import logging

import np
import torch
from utils.factory import get_model
import os
import datetime
import json



def train(args):
    _train(args)


def _train(args):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logs_name = "logs/{}/{}/{}_{}/{}".format(args["dataset"], args["model"], args["init_class"], args["increment"],
                                             now_time)
    workdir = os.path.join(logs_name, 'work_dir')
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    with open(os.path.join(workdir, 'configs.json'), 'w', encoding='utf-8') as json_file:
        json.dump(args, json_file, indent=2)

    logfilename = "logs/{}/{}/{}_{}/{}/{}_{}_{}_{}_{}".format(
        args["dataset"],
        args["model"],
        args["init_class"],
        args["increment"],
        now_time,
        args["dataset"],
        args["model"],
        args["init_class"],
        args["increment"],
        args["backbone_type"]
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
        ],
    )

    _set_device(args)
    print_args(args)

    model = get_model(args, logging)
    # model.init_train(data_loader, class_mask, data_loader_per_cls, target_task_map)
    # model.after_train(data_loader=data_loader,acc_matrix=acc_matrix,acc_matrix_accn=acc_matrix_accn)
    model.init_train()
    # model.cur_task += 1
    save_path = os.path.join(workdir, 'cub_lr0.001_epoch50_bs64.pth')
    # save_path = os.path.join(
    #     '/home/shengqin/ssd2ttt/fxr_new/MiN(fine-tuning backbone+crs+task_prompt+KD)/MiN/logs-new/cub/min/100_10/2026-04-18-17-45-26/work_dir/cub_lr0.001-epoch30_bs64_adam.pth')
    model.save_backbone_checkpoint(save_path)
    # model.load_backbone_checkpoint(save_path)
    model.train()
    model.after_train()
    # save_path = os.path.join(workdir, 'task_0_check_point.pth')
    # if args["save_all_checkpoint"] is True:
    #     model.save_check_point(save_path)

    for i in range(args['sessions']-1):
        # save_path = os.path.join(workdir, 'task_{}_check_point.pth'.format(i+1))
        # model.increment_train(data_loader, class_mask, data_loader_per_cls,
        #                       target_task_map,acc_matrix=pre_ca_acc_matrix,
        #                       acc_matrix_accn=pre_ca_acc_matrix_accn)
        # model.after_train(data_loader=data_loader,acc_matrix=acc_matrix,acc_matrix_accn=acc_matrix_accn)
        # save_path = os.path.join(workdir, 'cub_lr0.001_epoch30_bs64.pth')
        # save_path = os.path.join('/home/shengqin/ssd2ttt/fxr_new/MiN(fine-tuning backbone+crs+task_prompt)/MiN/logs-new/cifar224/min/60_5/2026-04-22-01-10-29/work_dir/cifar_lr0.001-epoch30_bs128_adam.pth')
        # model.load_backbone_checkpoint(save_path)
        model.cur_task += 1
        model.train()
        model.after_train()
        # if args["save_all_checkpoint"] is True:
        #     model.save_check_point(save_path)
    save_path = os.path.join(logs_name, 'Last_check_point.pth')
    model.save_check_point(save_path)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)

    args["device"] = gpus[0]


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _set_random(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
