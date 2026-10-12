import numpy as np
import torch
from dataloader.sampler import CategoriesSampler
from torch.utils.data import Dataset


def set_up_datasets(args):
    if args['dataset'] == 'cifar224':
        import dataloader.cifar100.cifar_vit as Dataset
        args['init_class'] = 60
        args['num_classes'] = 100
        args['way'] = 5
        args['shot'] = 5
        args['sessions'] = 9

    if args['dataset'] == 'cub':
        import dataloader.cub200.cub200_vit as Dataset
        args['init_class'] = 100
        args['num_classes'] = 200
        args['way'] = 10
        args['shot'] = 5
        args['sessions'] = 11

    if args['dataset'] == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet_vit as Dataset
        args['init_class'] = 60
        args['num_classes'] = 100
        args['way'] = 5
        args['shot'] = 5
        args['sessions'] = 9

    if args['dataset'] == 'imagenet_100':
        import dataloader.imagenet_100.imagenet_100 as Dataset
        args['init_class'] = 20
        args['num_classes'] = 100
        args['way'] = 8
        args['shot'] = 5
        args['sessions'] = 11

    if args['dataset'] == 'imagenet-r':
        import dataloader.imagenetr.imagenetr_vit as Dataset
        args['init_class'] = 100
        args['num_classes'] = 200
        args['way'] = 10
        args['shot'] = 5
        args['sessions'] = 11

    if args['dataset'] == 'imagenet-a':
        import dataloader.imageneta.imageneta_vit as Dataset
        args['init_class'] = 100
        args['num_classes'] = 200
        args['way'] = 10
        args['shot'] = 5
        args['sessions'] = 11

    if args['dataset'] == 'tiny_imagenet':
        import dataloader.Tiny_imagenet.Tiny_imagenet as Dataset
        args['init_class'] = 100
        args['num_classes'] = 200
        args['way'] = 10
        args['shot'] = 5
        args['sessions'] = 11

    if args['dataset'] == 'vtab':
        import dataloader.vtab.vtab_dinov3_b as Dataset
        args['init_class'] = 14
        args['num_classes'] = 50
        args['way'] = 4
        args['shot'] = 5
        args['sessions'] = 10

    args['Dataset'] = Dataset
    return args


def get_dataloader(args, session):
    set_up_datasets(args)
    if session == 0:
        trainset, trainloader, testloader, per_session_testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader, per_session_testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader, per_session_testloader


def get_base_dataloader(args):
    txt_path = "data/index_list/" + args['dataset'] + "/session_" + str(0) + '.txt'
    class_index = np.arange(args['base_class'])

    if args['dataset'] == 'cifar224':
        trainset = args['Dataset'].CIFAR100(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=True,
            download=True,
            index=class_index,
            base_sess=True
        )
        testset = args['Dataset'].CIFAR100(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            download=False,
            index=class_index,
            base_sess=True
        )

    if args['dataset'] == 'imagenet-a':
        trainset = args['Dataset'].ImageNet_A(
            root='/home/shengqin/fxr_new/datasets',
            train=True,
            index=class_index,
            base_sess=True
        )
        testset = args['Dataset'].ImageNet_A(
            root='/home/shengqin/fxr_new/datasets',
            train=False,
            index=class_index,
            base_sess=True
        )

    if args['dataset'] == 'cub':
        trainset = args['Dataset'].CUB200(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=True,
            index=class_index,
            base_sess=True
        )
        testset = args['Dataset'].CUB200(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            index=class_index
        )

    if args['dataset'] == 'mini_imagenet':
        trainset = args['Dataset'].MiniImageNet(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=True,
            index=class_index,
            base_sess=True
        )
        testset = args['Dataset'].MiniImageNet(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            index=class_index
        )

    if args['dataset'] == 'imagenet_100':
        trainset = args['Dataset'].ImageNet100(
            root='/home/shengqin/fxr_new/datasets',
            train=True,
            index=class_index,
            base_sess=True
        )
        testset = args['Dataset'].ImageNet100(
            root='/home/shengqin/fxr_new/datasets',
            train=False,
            index=class_index
        )

    if args['dataset'] == 'tiny_imagenet':
        trainset = args['Dataset'].Tiny_Imagenet(
            root='/home/shengqin/fxr_new/datasets/tiny-imagenet-200',
            train=True,
            index=class_index,
            base_sess=True
        )
        testset = args['Dataset'].Tiny_Imagenet(
            root='/home/shengqin/fxr_new/datasets/tiny-imagenet-200',
            train=False,
            index=class_index
        )

    if args['dataset'] == 'imagenet-r':
        trainset = args['Dataset'].ImageNet_R(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=True,
            index=class_index,
            base_sess=True
        )
        testset = args['Dataset'].ImageNet_R(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            index=class_index
        )

    if args['dataset'] == 'vtab':
        trainset = args['Dataset'].VTAB(
            root='/home/shengqin/fxr_new/datasets',
            train=True,
            index=class_index,
            base_sess=True
        )
        testset = args['Dataset'].VTAB(
            root='/home/shengqin/fxr_new/datasets',
            train=False,
            index=class_index
        )

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args['batch_size_base'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args['test_batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    per_session_testloader = testloader

    return trainset, trainloader, testloader, per_session_testloader


def get_new_dataloader(args, session):
    txt_path = "data/index_list/" + args['dataset'] + "/session_" + str(session) + '.txt'

    if args['dataset'] == 'cifar224':
        class_index = open(txt_path).read().splitlines()
        trainset = args['Dataset'].CIFAR100(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=True,
            download=False,
            index=class_index,
            base_sess=False
        )

    if args['dataset'] == 'cub':
        trainset = args['Dataset'].CUB200(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=True,
            index_path=txt_path
        )

    if args['dataset'] == 'mini_imagenet':
        trainset = args['Dataset'].MiniImageNet(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=True,
            index_path=txt_path
        )

    if args['dataset'] == 'imagenet_100':
        trainset = args['Dataset'].ImageNet100(
            root='/home/shengqin/fxr_new/datasets',
            train=True,
            index_path=txt_path
        )

    if args['dataset'] == 'imagenet-a':
        trainset = args['Dataset'].ImageNet_A(
            root='/home/shengqin/fxr_new/datasets',
            train=True,
            index_path=txt_path
        )

    if args['dataset'] == 'tiny_imagenet':
        trainset = args['Dataset'].Tiny_Imagenet(
            root='/home/shengqin/fxr_new/datasets/tiny-imagenet-200',
            train=True,
            index_path=txt_path
        )

    if args['dataset'] == 'imagenet-r':
        trainset = args['Dataset'].ImageNet_R(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=True,
            index_path=txt_path
        )

    if args['dataset'] == 'vtab':
        trainset = args['Dataset'].VTAB(
            root='/home/shengqin/fxr_new/datasets',
            train=True,
            index_path=txt_path
        )

    if args['batch_size_new'] == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=batch_size_new,
            shuffle=False,
            num_workers=args['num_workers'],
            pin_memory=True
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args['batch_size_new'],
            shuffle=True,
            num_workers=args['num_workers'],
            pin_memory=True
        )

    # test on this_session classes
    per_session_class = get_per_session_classes(args, session)

    if args['dataset'] == 'cifar224':
        per_testset = args['Dataset'].CIFAR100(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            download=False,
            index=per_session_class,
            base_sess=False
        )

    if args['dataset'] == 'cub':
        per_testset = args['Dataset'].CUB200(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            index=per_session_class
        )

    if args['dataset'] == 'mini_imagenet':
        per_testset = args['Dataset'].MiniImageNet(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            index=per_session_class
        )

    if args['dataset'] == 'imagenet_100':
        per_testset = args['Dataset'].ImageNet100(
            root='/home/shengqin/fxr_new/datasets',
            train=False,
            index=per_session_class
        )

    if args['dataset'] == 'imagenet-r':
        per_testset = args['Dataset'].ImageNet_R(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            index=per_session_class
        )

    if args['dataset'] == 'imagenet-a':
        per_testset = args['Dataset'].ImageNet_A(
            root='/home/shengqin/fxr_new/datasets',
            train=False,
            index=per_session_class
        )

    if args['dataset'] == 'tiny_imagenet':
        per_testset = args['Dataset'].Tiny_Imagenet(
            root='/home/shengqin/fxr_new/datasets/tiny-imagenet-200',
            train=False,
            index=per_session_class
        )

    if args['dataset'] == 'vtab':
        per_testset = args['Dataset'].VTAB(
            root='/home/shengqin/fxr_new/datasets',
            train=False,
            index=per_session_class
        )

    per_session_testloader = torch.utils.data.DataLoader(
        dataset=per_testset,
        batch_size=args['test_batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
        pin_memory=True
    )

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args['dataset'] == 'cifar224':
        testset = args['Dataset'].CIFAR100(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            download=False,
            index=class_new,
            base_sess=False
        )

    if args['dataset'] == 'cub':
        testset = args['Dataset'].CUB200(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            index=class_new
        )

    if args['dataset'] == 'mini_imagenet':
        testset = args['Dataset'].MiniImageNet(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            index=class_new
        )

    if args['dataset'] == 'imagenet_100':
        testset = args['Dataset'].ImageNet100(
            root='/home/shengqin/fxr_new/datasets',
            train=False,
            index=class_new
        )

    if args['dataset'] == 'imagenet-r':
        testset = args['Dataset'].ImageNet_R(
            root='/home/shengqin/fxr/new_caprompt/few_shot/caprompt-fscil_dataset/datasets',
            train=False,
            index=class_new
        )

    if args['dataset'] == 'imagenet-a':
        testset = args['Dataset'].ImageNet_A(
            root='/home/shengqin/fxr_new/datasets',
            train=False,
            index=class_new
        )

    if args['dataset'] == 'tiny_imagenet':
        testset = args['Dataset'].Tiny_Imagenet(
            root='/home/shengqin/fxr_new/datasets/tiny-imagenet-200',
            train=False,
            index=class_new
        )

    if args['dataset'] == 'vtab':
        testset = args['Dataset'].VTAB(
            root='/home/shengqin/fxr_new/datasets',
            train=False,
            index=class_new
        )

    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args['test_batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
        pin_memory=True
    )

    return trainset, trainloader, testloader, per_session_testloader


def get_session_classes(args, session):
    class_list = np.arange(args['base_class'] + session * args['way'])
    return class_list


def get_per_session_classes(args, session):
    start = args['init_class'] + (session - 1) * args['way']
    end = start + args['way']
    class_list = np.arange(start, end)
    return class_list