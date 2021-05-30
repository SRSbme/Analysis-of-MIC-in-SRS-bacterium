from tqdm import tqdm
import argparse
from utils.loss import SegmentationLosses
from model import Model
import os
import torch
from torch.utils.data import DataLoader
from utils.dataset import OwnDataset
from utils.metrics import dice_coeff, mean_accuracy, mean_IU
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', metavar='E', type=int, default=50, help='Number of epochs', dest='epochs')
    parser.add_argument('--train_batch_size', metavar='B', type=int, nargs='?', default=8, help='Train Batch size',
                        dest='train_batch_size')
    parser.add_argument('--val_batch_size', metavar='B', type=int, nargs='?', default=1, help='Val Batch size',
                        dest='val_batch_size')
    parser.add_argument('--learning-rate', metavar='LR', type=float, nargs='?', default=0.001, help='Learning rate',
                        dest='lr')
    parser.add_argument('--net', dest='net', type=str, default='UNet',
                        choices=['Improved_UNet', 'UNet', 'MultiUnet', 'deeplabv3plus', 'fcn', 'maskrcnn', 'CascadePSP',
                                 'UNet++'])
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkpoint', type=str, default='checkpoint', help='save model path')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--loss_type', dest='loss_type', type=str, default='ce', choices=['ce', 'focal', 'SoftDice'])
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.net = args.net
        self.resume = args.resume
        self.epochs = args.epochs
        self.lose_type = args.loss_type
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.lr = args.lr
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_pred = 0.0
        self.checkpoint = os.path.join(args.checkpoint, self.net)
        os.makedirs(self.checkpoint, exist_ok=True)

        train_dataset = OwnDataset('data', 'train.txt')
        eval_dataset = OwnDataset('data', 'val.txt')
        self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=0,
                                       pin_memory=True)
        self.val_loader = DataLoader(eval_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=0,
                                     pin_memory=True)
        self.model = Model(self.net)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 5, eta_min=0.00001, last_epoch=-1)
        self.criterion = SegmentationLosses(cuda=self.cuda).build_loss(lose_type=self.lose_type)

        if self.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            print('---------------------权重恢复中' + self.resume + '---------------------')
            self.model.load_state_dict(torch.load(self.resume, map_location=self.device))
            print('---------------------权重恢复完成---------------------')

        # 将模型加载进gpu
        if self.cuda:
            sum_gpu = torch.cuda.device_count()
            print('可使用gpu共' + str(sum_gpu) + '个' + torch.cuda.get_device_name(0))
            gpu_ids = [s for s in range(sum_gpu)]
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)  # device_ids不写，默认有几个gpu就用几个,键会加module
            self.model.cuda()
            self.model.to(self.device)

    def train(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        item = 0
        for image, target in tbar:
            image = image.unsqueeze(dim=1)
            target = target.unsqueeze(dim=1)
            image = Variable(image.to(device=self.device, dtype=torch.float32))
            target = Variable(target.to(device=self.device, dtype=torch.float32))
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Epoch: %d Train loss: %.3f lr: %.10f' % (
                epoch + 1, train_loss / (item + 1), self.optimizer.state_dict()['param_groups'][0]['lr']))
            item = item + 1
        return train_loss / item

    def val(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        ###########分割指标###############
        ###########Pixel Accuracy(PA，像素精度)：这是最简单的度量，为标记正确的像素占总像素的比例。
        ###########Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        ###########Mean Intersection over Union(MIoU，均交并比)：为语义分割的标准度量。其计算两个集合的交集和并集之比，在语义分割的问题中，
        # 这两个集合为真实值（ground truth）和预测值（predicted segmentation）。这个比例可以变形为正真数（intersection）比上真正、假负、假正
        # （并集）之和。在每个类上计算IoU，之后平均。
        val_loss = 0.0
        val_dice = 0.0
        val_mpa = 0.0
        val_miou = 0.0
        item = 0
        for image_path, image, target in tbar:
            image = image.unsqueeze(dim=1)
            target = target.unsqueeze(dim=1)
            image = Variable(image.to(device=self.device, dtype=torch.float32))
            target = Variable(target.to(device=self.device, dtype=torch.float32))
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            val_loss += loss.item()
            pred = output.data.squeeze().cpu().numpy()
            target = target.squeeze().cpu().numpy()
            dice = dice_coeff(pred, target)
            mpa = mean_accuracy(pred, target)  # (584, 565) (584, 565)
            miou = mean_IU(pred, target)
            val_dice += dice
            val_mpa += mpa
            val_miou += miou
            tbar.set_description('Epoch: %d Val loss: %.3f Val dice: %.3f Val mpa: %.3f Val miou: %.3f lr: %.10f' % (
                epoch + 1, val_loss / (item + 1), val_dice / (item + 1), val_mpa / (item + 1), val_miou / (item + 1),
                self.optimizer.state_dict()['param_groups'][0]['lr']))
            item = item + 1

        ################每次保存最好的模型###################
        if self.best_pred <= round(val_dice / item, 3):
            self.best_pred = round(val_dice / item, 3)
            if self.cuda:
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.checkpoint, self.net + f'_best_dice:{self.best_pred}.pth'))
            else:
                torch.save(self.model.state_dict(),
                           os.path.join(self.checkpoint, self.net + f'_best_dice:{self.best_pred}.pth'))

        return val_loss / item, val_dice / item, val_mpa / item, val_miou / item

    def training(self):
        epoch_train_loss = []
        epoch_val_loss = []
        epoch_val_dice = []
        epoch_val_mpa = []
        epoch_val_miou = []
        logs_dir = os.path.join('results', self.net, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        for epoch in range(self.epochs):
            train_loss = self.train(epoch)
            self.scheduler.step()
            val_loss, val_dice, val_mpa, val_miou = self.val(epoch)
            epoch_train_loss.append(train_loss)
            epoch_val_loss.append(val_loss)
            epoch_val_dice.append(val_dice)
            epoch_val_mpa.append(val_mpa)
            epoch_val_miou.append(val_miou)

            x = list(range(len(epoch_train_loss)))
            fig, ax = plt.subplots()
            ax.plot(np.array(x), np.array(epoch_train_loss), label='train_loss')
            ax.plot(np.array(x), np.array(epoch_val_loss), label='val_loss')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title('epoch-loss')
            ax.legend()
            plt.title("epoch-loss")
            plt.savefig(os.path.join(logs_dir, 'loss.png'))
            plt.close(1)

            fig, ax = plt.subplots()
            ax.plot(np.array(x), np.array(epoch_val_dice), label='val_dice')
            ax.plot(np.array(x), np.array(epoch_val_mpa), label='val_mpa')
            ax.plot(np.array(x), np.array(epoch_val_miou), label='val_miou')
            ax.set_xlabel('epoch')
            ax.set_ylabel('acc')
            ax.set_title('epoch-acc')
            ax.legend()
            plt.title("epoch-acc")
            plt.savefig(os.path.join(logs_dir, 'acc.png'))
            plt.close(1)


def main():
    arg = get_args()
    print(arg)
    trainer = Trainer(arg)
    trainer.training()


if __name__ == "__main__":
    main()

