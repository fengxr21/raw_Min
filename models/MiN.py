import math

import sys

import numpy as np
import torch.distributed as dist
from sympy.codegen.ast import none
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import torch
from torch import optim, no_grad
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import MiNbaseNet
from utils.toolkit import tensor2numpy
from dataloader.data_utils import get_dataloader
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics

import utils_1

EPSILON = 1e-8


class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = args["num_workers"]

        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_batch_size = args["init_batch_size"]

        self.lr = args["lr"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.epochs = args["epochs"]
        self.init_epochs_base = args['init_epochs_base']

        self.init_class = args["init_class"]
        self.increment = args["increment"]

        self.buffer_size = args["buffer_size"]
        self.buffer_batch = args["buffer_batch"]
        self.gamma = args['gamma']
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        self.class_acc = []
        self.task_acc = []
        self.total_acc_crs = []

    def after_train(self):
        _, _, test_loader, per_session_test_loader = get_dataloader(self.args, self.cur_task)
        # eval_res = self.eval_task(test_loader)
        # current_task_acc = self.compute_current_task_acc(per_session_test_loader)
        current_task_acc = self.evaluate_reconstruction_all_seen(per_session_test_loader)
        recon_all_seen_acc = self.evaluate_reconstruction_all_seen(test_loader)
        self.task_acc.append(round(float(current_task_acc), 2))
        self.total_acc_crs.append(round(float(recon_all_seen_acc), 2))

        HAcc, old_acc, new_acc, per_task_accuracy = self.compute_hacc()

        self.logger.info('current_task_crs_acc: {}'.format(self.task_acc))
        self.logger.info('total acc_crs: {}'.format(self.total_acc_crs))
        self.logger.info('recon all-seen acc: {:.2f}'.format(recon_all_seen_acc))
        self.logger.info('avg_acc_crs: {:.2f}'.format(np.mean(self.total_acc_crs)))

        print('current_task_acc: {}'.format(self.task_acc))
        print('total acc_crs: {}'.format(self.total_acc_crs))
        print('recon all-seen acc: {:.2f}'.format(recon_all_seen_acc))
        print('avg_acc_crs: {:.2f}'.format(np.mean(self.total_acc_crs)))

        if HAcc is None:
            self.logger.info('HAcc=None, old_acc={:.2f}, new_acc=None'.format(old_acc))
            print('HAcc=None, old_acc={:.2f}, new_acc=None'.format(old_acc))
        else:
            self.logger.info('HAcc={:.2f}, old_acc={:.2f}, new_acc={:.2f}'.format(HAcc, old_acc, new_acc))
            print('HAcc={:.2f}, old_acc={:.2f}, new_acc={:.2f}'.format(HAcc, old_acc, new_acc))

    def save_check_point(self, path_name):
        save_dict = {
            # "model_state_dict": self._network.state_dict(),
            # "enhanced_prototype_pool": self._network.enhanced_prototype_pool,
            # "teacher_prototype_pool": self._network.teacher_prototype_pool,
            "prototype_pool": self._network.prototype_pool,
            # "prompt_pool": self._network.prompt_pool
        }
        torch.save(save_dict, path_name)

    def save_backbone_checkpoint(self, path_name):
        torch.save(self._network.backbone.state_dict(), path_name)

    def load_backbone_checkpoint(self, path_name, strict=True):
        state_dict = torch.load(path_name, map_location=self.device)
        self._network.backbone.load_state_dict(state_dict, strict=strict)
        self._network.to(self.device)
        print("\nLoaded backbone\n")

    def compute_test_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        device = self.device
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def compute_current_task_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        device = self.device

        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                logits = outputs["logits"]

            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts == targets).sum().item()
            total += targets.size(0)

        return round(100.0 * correct / total, 2) if total > 0 else 0.0

    def compute_current_task_prototypes(self, data_loader):
        self._network.eval()
        self._network.to(self.device)

        all_features = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                features = self._network.extract_feature(inputs)
                # features = self._network.buffer(features)  # [B, D]

                all_features.append(features.cpu())
                all_targets.append(targets.cpu())

                del inputs, targets, features
                torch.cuda.empty_cache()

        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        class_ids = torch.unique(all_targets).tolist()
        class_ids = sorted(class_ids)

        prototypes = []
        for cid in class_ids:
            mask = (all_targets == cid)
            proto = all_features[mask].mean(dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=0).to(self.device)
        self._network.save_raw_prototypes(task_id=self.cur_task, class_ids=class_ids, prototypes=prototypes)
        return prototypes, class_ids

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    def build_train_prototype_dictionary(self):
        current_enhanced_prototypes, current_class_ids = \
            self._network.get_current_enhanced_prototypes(self.cur_task)

        pre_enhanced_prototypes, pre_class_ids = \
            self._network.get_pre_enhanced_prototypes(self.cur_task)

        if pre_enhanced_prototypes is None:
            all_enhanced_prototypes = current_enhanced_prototypes
            all_class_ids = current_class_ids
        else:
            all_enhanced_prototypes = torch.cat(
                [pre_enhanced_prototypes, current_enhanced_prototypes], dim=0
            )
            all_class_ids = pre_class_ids + current_class_ids

        return all_enhanced_prototypes, all_class_ids

    def map_targets_to_indices(self, targets, all_class_ids):
        class_to_idx = {cid: i for i, cid in enumerate(all_class_ids)}
        mapped_targets = torch.tensor(
            [class_to_idx[int(t.item())] for t in targets],
            dtype=torch.long,
            device=targets.device
        )
        return mapped_targets

    def init_train(self):
        self.cur_task += 1  # base task -> 0

        train_set, train_loader, test_loader, per_session_test_loader = get_dataloader(self.args, self.cur_task)

        self.logger.info("========== Base Session ==========")
        self.logger.info("Current task id: {}".format(self.cur_task))
        self.logger.info("Train samples: {}".format(len(train_set)))
        self.logger.info("All-seen test samples: {}".format(len(test_loader.dataset)))
        self.logger.info("Per-session test samples: {}".format(len(per_session_test_loader.dataset)))

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = True

        # base阶段扩展分类头
        self._network.update_fc(self.init_class)

        self.run(train_loader)

    def train(self):

        train_set, train_loader, test_loader, per_session_test_loader = \
            get_dataloader(self.args, self.cur_task)

        if not hasattr(self, "all_task_testloader"):
            self.all_task_testloader = []
        if len(self.all_task_testloader) <= self.cur_task:
            self.all_task_testloader.append(per_session_test_loader)

        self.logger.info("========== Train Session ==========")
        self.logger.info("Current task id: {}".format(self.cur_task))
        self.logger.info("Train samples: {}".format(len(train_set)))
        self.logger.info("All-seen test samples: {}".format(len(test_loader.dataset)))
        self.logger.info("Per-session test samples: {}".format(len(per_session_test_loader.dataset)))

        self._network.to(self.device)
        self._network.eval()
        for param in self._network.parameters():
            param.requires_grad = False
        n_parameters = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        print('Model number of params:', n_parameters)
        raw_prototypes, class_ids = self.compute_current_task_prototypes(train_loader)

        prog_bar = tqdm(range(self.init_epochs_base))

        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            correct, total = 0, 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                # print("before:", self._network.all_prompts[class_ids][:2, :5].detach().cpu())

                with torch.no_grad():
                    features = self._network.extract_feature(inputs)  # [B, D]
                    # features = self._network.buffer(features)
                all_enhanced_prototypes, all_class_ids = self.build_train_prototype_dictionary()
                if self.args['class_mode'] == 'crs':
                    logits = self.classify_with_crs_from_prototypes(all_enhanced_prototypes, features)
                elif self.args['class_mode'] == 'ed':
                    logits = self.classify_with_ed_from_prototypes(all_enhanced_prototypes, features)
                elif self.args['class_mode'] == 'square_ed':
                    logits = self.classify_with_square_ed_from_prototypes(all_enhanced_prototypes, features)
                elif self.args['class_mode'] == 'cosine':
                    logits = self.classify_with_cosine_from_prototypes(all_enhanced_prototypes, features)
                mapped_targets = self.map_targets_to_indices(targets, all_class_ids)
                preds = torch.max(logits, dim=1)[1]
                correct += (preds == mapped_targets).sum().item()
                total += mapped_targets.size(0)

            train_acc = 100.0 * correct / total if total > 0 else 0.0
            info = "Task {} Epoch {}/{}, Train_acc {:.2f}".format(
                self.cur_task, epoch + 1, self.epochs,  train_acc
            )
            self.logger.info(info)
            prog_bar.set_description(info)
        final_enhanced_prototypes, _ = self._network.get_current_enhanced_prototypes(self.cur_task)
        self._network.save_final_current_enhanced_prototypes(
            self.cur_task,
            class_ids,
            final_enhanced_prototypes
        )

    def fit_fc(self, train_loader):
        self._network.eval()
        self._network.to(self.device)

        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.to(self.device)
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, targets)

            info = "Task {} --> Update Analytical Classifier!".format(
                self.cur_task,
            )
            self.logger.info(info)
            prog_bar.set_description(info)

    def re_fit(self, train_loader):
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(train_loader)
        for i, (inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets)
            self._network.fit(inputs, targets)

            info = "Task {} --> Reupdate Analytical Classifier!".format(
                self.cur_task,
            )
            
            self.logger.info(info)
            prog_bar.set_description(info)

    def periodic_validate(self, fit_loader, val_loader):
        self._network.eval()
        self._network.to(self.device)
        # 备份当前解析解状态，避免污染正式训练后的解析分类器
        old_weight = self._network.weight.clone()
        old_R = self._network.R.clone()
        # 额外备份当前训练/评估状态，评估完恢复
        old_mode = self._network.training
        try:
            # 用当前backbone重新拟合一个临时解析解分类器
            self._network.reset_analytic_classifier()
            with torch.no_grad():
                for inputs, targets in fit_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    targets = torch.nn.functional.one_hot(
                        targets,
                        num_classes=self._network.known_class
                    )
                    self._network.fit(inputs, targets)
            # 在验证集上测试
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self._network(inputs)
                    logits = outputs["logits"]
                    if logits.shape[1] == 0:
                        return 0.0
                    predicts = torch.max(logits, dim=1)[1]
                    correct += (predicts == targets).sum().item()
                    total += targets.size(0)
            val_acc = round(100.0 * correct / total, 2) if total > 0 else 0.0
        finally:
            # 恢复原来的解析解状态
            self._network.weight = old_weight
            self._network.R = old_R
            # 恢复模型模式
            if old_mode:
                self._network.train()
            else:
                self._network.eval()
        return val_acc

    def evaluate_reconstruction_all_seen(self, test_loader):
        self._network.eval()
        self._network.to(self.device)

        old_mode = self._network.training
        try:
            all_prototypes, all_class_ids = self._network.get_all_previous_prototypes()
            if all_prototypes is None or len(all_class_ids) == 0:
                return 0.0
            proto_class_ids = torch.tensor(
                all_class_ids,
                device=self.device,
                dtype=torch.long
            )
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    features = self._network.extract_feature(inputs)
                    # features = self._network.buffer(features)
                    all_enhanced_prototypes, all_class_ids = self.build_train_prototype_dictionary()
                    if self.args['class_mode'] == 'crs':
                        logits = self.classify_with_crs_from_prototypes(all_enhanced_prototypes, features)
                    elif self.args['class_mode'] == 'ed':
                        logits = self.classify_with_ed_from_prototypes(all_enhanced_prototypes, features)
                    elif self.args['class_mode'] == 'square_ed':
                        logits = self.classify_with_square_ed_from_prototypes(all_enhanced_prototypes, features)
                    elif self.args['class_mode'] == 'cosine':
                        logits = self.classify_with_cosine_from_prototypes(all_enhanced_prototypes, features)
                    pred_idx = torch.argmax(logits, dim=1)
                    predicts = proto_class_ids[pred_idx]
                    correct += (predicts == targets).sum().item()
                    total += targets.size(0)
            recon_acc = round(100.0 * correct / total, 2) if total > 0 else 0.0
        finally:
            if old_mode:
                self._network.train()
            else:
                self._network.eval()
        return recon_acc

    def evaluate_loader_acc(self, data_loader):
        self._network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                features = self._network.extract_feature(inputs)
                # features = self._network.buffer(features)
                all_enhanced_prototypes, proto_class_ids = self.build_train_prototype_dictionary()
                proto_class_ids = torch.tensor(
                    proto_class_ids,
                    device=self.device,
                    dtype=torch.long
                )

                if self.args['class_mode'] == 'crs':
                    logits = self.classify_with_crs_from_prototypes(all_enhanced_prototypes, features)
                elif self.args['class_mode'] == 'ed':
                    logits = self.classify_with_ed_from_prototypes(all_enhanced_prototypes, features)
                elif self.args['class_mode'] == 'square_ed':
                    logits = self.classify_with_square_ed_from_prototypes(all_enhanced_prototypes, features)
                elif self.args['class_mode'] == 'cosine':
                    logits = self.classify_with_cosine_from_prototypes(all_enhanced_prototypes, features)

                pred_idx = torch.argmax(logits, dim=1)
                predicts = proto_class_ids[pred_idx]

                correct += (predicts == targets).sum().item()
                total += targets.size(0)

        acc = 100.0 * correct / total if total > 0 else 0.0
        return acc

    def compute_hacc(self):
        per_task_accuracy = {}

        for task_id, task_testloader in enumerate(self.all_task_testloader[:self.cur_task + 1]):
            task_acc = self.evaluate_loader_acc(task_testloader)
            per_task_accuracy[f"task_{task_id + 1}"] = round(float(task_acc), 2)

        old_acc = per_task_accuracy.get("task_1", 0.0)

        if self.cur_task >= 1:
            new_acc_list = [
                per_task_accuracy[f"task_{task_id + 1}"]
                for task_id in range(1, self.cur_task + 1)
            ]
            new_acc = sum(new_acc_list) / len(new_acc_list) if len(new_acc_list) > 0 else 0.0
            HAcc = (2 * old_acc * new_acc) / (old_acc + new_acc) if (old_acc + new_acc) > 0 else 0.0
        else:
            new_acc = None
            HAcc = None

        return HAcc, old_acc, new_acc, per_task_accuracy

    def run(self, train_loader):
        if self.cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = self.epochs
            lr = self.lr
            weight_decay = self.weight_decay

        for param in self._network.parameters():
            param.requires_grad = False
        for param in self._network.normal_fc.parameters():
            param.requires_grad = True

        if self.cur_task == 0:
            self._network.init_unfreeze()
        # self._network.init_unfreeze()

        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0

            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.cur_task > 0:
                    with torch.no_grad():
                        outputs1 = self._network(inputs)
                        logits1 = outputs1['logits']

                    outputs2 = self._network.forward_normal_fc(inputs)
                    logits2 = outputs2['logits']
                    logits2 = logits2 + logits1
                    loss = F.cross_entropy(logits2, targets.long())
                    _, preds = torch.max(logits2, dim=1)
                else:
                    outputs = self._network.forward_normal_fc(inputs)
                    logits = outputs["logits"]
                    loss = F.cross_entropy(logits, targets.long())
                    _, preds = torch.max(logits, dim=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = 100. * correct / total

            base_info = "Task {} --> Learning Beneficial Noise!: Epoch {}/{} => Loss {:.3f}, train_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )

            prog_bar.set_description(base_info)

            self.logger.info(base_info)
            prog_bar.set_description(base_info)

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)
            logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            pred.extend([int(predicts[i].cpu().numpy()) for i in range(predicts.shape[0])])
            label.extend(int(targets[i].cpu().numpy()) for i in range(targets.shape[0]))
        class_info = calculate_class_metrics(pred, label)
        task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {
            "all_class_accy": class_info['all_accy'],
            "class_accy": class_info['class_accy'],
            "class_confusion": class_info['class_confusion_matrices'],
            "task_accy": task_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
            "all_task_accy": task_info['task_accy'],
        }

    def get_task_prototype(self, model, train_loader):
        model = model.eval()
        model.to(self.device)
        features = []
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                feature = model.extract_feature(inputs)
                # features = self._network.buffer(features)
            features.append(feature)
        prototype = torch.mean(torch.concat(features, dim=0), dim=0)
        return prototype

    def classify_with_ed_from_prototypes(self, prototypes, features, squared=False):
        prototypes = prototypes.to(features.device)
        prototypes = prototypes.to(features.dtype)
        dists = torch.cdist(features, prototypes, p=2)

        if squared:
            dists = dists ** 2

        logits = -dists
        return logits

    def classify_with_square_ed_from_prototypes(self, prototypes, features):
        x = features.unsqueeze(1)
        y = prototypes.unsqueeze(0)
        dists = torch.pow(x - y, 2).mean(2)
        logits = -dists
        return logits

    def classify_with_cosine_from_prototypes(self, prototypes, features):
        prototypes = prototypes.to(features.device)
        prototypes = prototypes.to(features.dtype)

        features_norm = features / features.norm(dim=1, keepdim=True).clamp(min=1e-8)
        prototypes_norm = prototypes / prototypes.norm(dim=1, keepdim=True).clamp(min=1e-8)

        logits = torch.mm(features_norm, prototypes_norm.t())

        return logits

    def classify_with_crs_from_prototypes(self, prototypes, features, eps=1e-8):
        # [B, C, D]
        proto_expand = prototypes.unsqueeze(0).expand(features.size(0), -1, -1)
        # [B, D, C]
        X = proto_expand.transpose(-1, -2)
        # [B, C, C]
        XtX = torch.bmm(X.transpose(-2, -1), X)
        C = prototypes.size(0)
        I = torch.eye(C, device=features.device, dtype=features.dtype).unsqueeze(0)
        lambda_crs = self.args.get("lambda_crs", self.gamma)
        XtX_in = XtX + lambda_crs * I
        # [B, C]
        Xty = torch.bmm(X.transpose(-2, -1), features.unsqueeze(-1)).squeeze(-1)
        # [B, C]
        rho = torch.linalg.solve(XtX_in, Xty)
        # [B, C, D]
        recon = rho.unsqueeze(-1) * proto_expand
        # [B, C, D]
        diff = features.unsqueeze(1) - recon
        # [B, C]
        residuals_unnorm = torch.norm(diff, dim=-1)
        norm_rho = torch.norm(rho, dim=-1, keepdim=True) + eps
        residuals = residuals_unnorm / norm_rho

        return -residuals
            
def norm(feature):
    feature_norm = torch.sum(feature,dim=-1)
    feature = feature/feature_norm.unsqueeze(-1)
    return feature


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
