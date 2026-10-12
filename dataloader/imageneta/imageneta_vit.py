import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .autoaugment import AutoAugImageNetPolicy


class ImageNet_A(Dataset):
    def __init__(self, root='', train=True, transform=None, index_path=None, index=None, base_sess=None, autoaug=False):
        if train:
            setname = 'train'
        else:
            setname = 'test'

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.base_sess = base_sess
        self.autoaug = autoaug
        self.IMAGE_PATH = os.path.join(self.root, 'imagenet-a', setname)  # ← changed
        assert os.path.exists(self.IMAGE_PATH), f"Image path not found: {self.IMAGE_PATH}"

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1
        self.wnids = []

        # Traverse class directories in sorted order to ensure consistent labeling
        for wnid in sorted(os.listdir(self.IMAGE_PATH)):
            class_dir = osp.join(self.IMAGE_PATH, wnid)
            if not os.path.isdir(class_dir):
                continue
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                path = osp.join(class_dir, img_name)
                self.data.append(path)
                self.targets.append(lb)
                self.data2label[path] = lb

        if autoaug is False:
            # do not use autoaug.
            if train:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ])
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        else:
            # use autoaug.
            if train:
                image_size = 224  # 84
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    # add autoaug
                    AutoAugImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                image_size = 224  # 84
                self.transform = transforms.Compose([
                    transforms.Resize(256),  # [92,92]
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        self.original_length = len(self.data)

    def SelectfromTxt(self, data2label, index_path):
        """Select from txt file. Each line should be: 'wnid/filename.JPEG'"""
        index = []
        with open(index_path, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
        for line in lines:
            # line example: "n01440764/n01440764_123.JPEG"
            parts = line.split('/')
            rel_path = '/'.join(parts[-2:])
            if len(parts) >= 2:
                index.append(rel_path)  # keep full relative path

        data_tmp = []
        targets_tmp = []
        for rel_path in index:
            full_path = osp.join(self.IMAGE_PATH, rel_path)
            if full_path in data2label:
                data_tmp.append(full_path)
                targets_tmp.append(data2label[full_path])
            else:
                # fallback: try direct path (if txt contains absolute or full relative)
                if rel_path in data2label:
                    data_tmp.append(rel_path)
                    targets_tmp.append(data2label[rel_path])
        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        """Select all instances from given class indices"""
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(np.array(targets) == i)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)
        # if not self.base_sess and self.train:
        #     return self.original_length * 4
        # else:
        #     return self.original_length

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets
        # img_path, target = self.data[index], self.targets[index]
        # # 使用Image.open()打开图像而不是Image.fromarray()
        # img = Image.open(img_path).convert('RGB')
        # # 使用预先加载的processor处理图像
        # inputs = self.processor(images=img, return_tensors="pt")
        # img = inputs['pixel_values'].squeeze(0)
        # return img, target
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target
        # if not self.base_sess and self.train:
        #     # 映射到原始索引和旋转次数
        #     orig_idx = i // 4
        #     rot_k = i % 4
        # else:
        #     orig_idx = i
        #     rot_k = 0
        # path = self.data[orig_idx]
        # img = Image.open(path).convert('RGB')
        # # 执行旋转（PIL.rotate 是逆时针，所以用 -rot_k*90 实现顺时针）
        # if rot_k != 0:
        #     img = img.rotate(-rot_k * 90, expand=False)
        # if self.transform:
        #     img = self.transform(img)
        # label = self.targets[orig_idx]
        # return img, label


# Optional: Concatenated dataset (same as original)
class Imagenet_R_concate(Dataset):
    def __init__(self, train, x1, y1, x2, y2):
        image_size = 84
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.data = x1 + x2
        self.targets = y1 + y2
        print(len(self.data), len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, target = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, target