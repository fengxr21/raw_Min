import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

class VTAB(Dataset):
    def __init__(self, root='', train=True,
                 index_path=None, index=None, base_sess=None, autoaug=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.base_sess = base_sess

        self.original_classes = list(range(10, 60))
        self.class_to_label = {cid: i for i, cid in enumerate(self.original_classes)}

        self.data = []
        self.targets = []

        self._load_all_data()

        if train:
            if base_sess:
                assert index is not None
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                assert index_path is not None
                self.data, self.targets = self.SelectfromTxt(index_path)
        else:
            assert index is not None
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        model_path = '/home/shengqin/fxr/new_caprompt/few_shot/full_enhanced_prototype/dino-v2_enhanced_prototype_fscil_dataset/model/dinov2_L'
        # model_path = '/home/shengqin/fxr/new_caprompt/few_shot/full_enhanced_prototype/dino-v2_enhanced_prototype_fscil_dataset/model/dinov2'
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)

    def _load_all_data(self):
        folder_name = 'train' if self.train else 'test'
        # 关键：加上 'vtab' 子目录
        data_dir = os.path.join(self.root, 'vtab', folder_name)
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory {data_dir} does not exist!")

        for orig_cls in self.original_classes:
            cls_folder = os.path.join(data_dir, str(orig_cls))
            if not os.path.isdir(cls_folder):
                continue
            label = self.class_to_label[orig_cls]
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_folder, img_name)
                    self.data.append(img_path)
                    self.targets.append(label)

    def SelectfromTxt(self, index_path):
        with open(index_path, 'r') as f:
            lines = f.read().splitlines()
        data_tmp = []
        targets_tmp = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                img_path, label = parts[0], int(parts[1])
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.root, img_path)
                # 可选：检查文件是否存在
                if not os.path.isfile(img_path):
                    print(f"⚠️ Warning: {img_path} not found, skipped.")
                    continue
                data_tmp.append(img_path)
                targets_tmp.append(label)
        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        index = np.array(index)
        data_tmp = []
        targets_tmp = []
        targets = np.array(targets)
        for cls in index:
            idxs = np.where(targets == cls)[0]
            for j in idxs:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        target = self.targets[i]
        image = Image.open(path).convert('RGB')
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].squeeze(0)
        return pixel_values, target