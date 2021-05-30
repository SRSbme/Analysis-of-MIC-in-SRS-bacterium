import argparse
from model import Model
import os
import torch
import cv2
import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt
from utils.watershed import watershed
from utils.draw_color import label_draw_color, img_draw_color, channel_single2three, draw_outer_circle


def save_img_mask(img, mask, name='tem.png'):
    plt.figure(dpi=300)
    plt.suptitle(name)
    plt.imshow(img, 'gray')
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask[:, :, 2] = 0
    mask[:, :, 0] = 0
    plt.imshow(mask, alpha=0.25, cmap='rainbow')
    plt.savefig(name)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--net', dest='net', type=str, default='fcn',
                        choices=['Improved_UNet', 'UNet', 'MultiUnet', 'deeplabv3plus', 'fcn', 'UNet++'])
    parser.add_argument('--resume', type=str, default='checkpoint/fcn/best_dice_0.791.pth',
                        help='put the path to resuming file if needed')
    parser.add_argument('--watershed', default=True, help='do you need to use watershed algorithm')
    return parser.parse_args()


arg = get_args()
model = Model(arg.net)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if arg.resume is None:
    arg.resume = os.path.join('checkpoint', arg.net, os.listdir('checkpoint/' + arg.net)[0])
model.load_state_dict(torch.load(arg.resume, map_location=device))
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    model = model.cuda()
model = model.to(device)
print('---------------------权重恢复完成---------------------')
model.eval()


# ./test_sets/data_xx/image/Step size0Dwell time50 A.b 124 Cefepime 1ug 852nm 25mw 250mw tune60.4 timing2420 stepsize 0.003 003.tif
def inference1():
    # 实现一张一张图的测试
    while True:
        img_path = input('Input image filename:')
        try:
            image = cv2.imread(img_path, -1)
        except:
            print('Open Error! Try again!')
            continue
        else:
            h, w = image.shape[0:2]
            img = image / 255.0
            img = cv2.resize(img, (256, 256))
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)
            img = img.unsqueeze(0)
            img = img.to(device=device, dtype=torch.float32)
            label = model(img)
            label = label.squeeze()
            label = label.cpu().detach().numpy()
            label = np.where(label > 0.5, 255, 0)
            label = label.astype(np.uint8)
            label = cv2.resize(label, (w, h))

            label_color = label_draw_color(label)
            img_color = img_draw_color(image, label)
            image_outer_circle = draw_outer_circle(copy.copy(image), label)

            if arg.watershed == True:
                watershed_label = np.expand_dims(label, 0).repeat(3, axis=0)
                watershed_label = np.transpose(watershed_label, [1, 2, 0])
                mask = watershed(watershed_label)
                mask = mask[:, :, 0]
                mask_color = label_draw_color(mask)
                image_mask_outer_circle = draw_outer_circle(copy.copy(image), mask)
                out = np.hstack((channel_single2three(image), channel_single2three(label), channel_single2three(mask),
                                 img_color, label_color, channel_single2three(mask_color), image_outer_circle,
                                 image_mask_outer_circle))
            else:
                # out = np.hstack((image, label))  # 同时可视化原图和测试结果图
                out = np.hstack((channel_single2three(image), channel_single2three(label), img_color, label_color,
                                 image_outer_circle))

            cv2.imshow(img_path, out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def inference2():
    # 实现图的批量测试
    dataset = 'data_xx'
    # dataset = '100images_filered'
    data_path_dir = os.path.join('test_sets', dataset, 'images')  # 需要测试的图片文件夹
    data_path = os.listdir(data_path_dir)
    for i in range(len(data_path)):
        data_path[i] = os.path.join(data_path_dir, data_path[i])
    # print(data_path)
    print('一共有' + str(len(data_path)) + '个图片待处理')
    save_dir = os.path.join('test_sets', dataset, 'results', arg.net)  # 定义需要保存测试结果的文件夹
    os.makedirs(save_dir, exist_ok=True)
    for data in data_path:
        print(data + '文件处理中')

        ori_img = Image.open(data).convert('L')  # 'L'转灰度图
        w, h = ori_img.size
        img = ori_img.resize((256, 256))
        img = np.array(img).astype(np.float32)
        img /= 255.0

        img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        label = model(img)
        label = label.squeeze()
        label = label.cpu().detach().numpy()
        label = np.where(label > 0.5, 255, 0)
        label = label.astype(np.uint8)
        label = cv2.resize(label, (w, h))

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # label = cv2.erode(label, kernel)
        label = cv2.erode(label, None, iterations=1)

        ori_img = np.asarray(ori_img)

        label_color = label_draw_color(label)
        img_color = img_draw_color(ori_img, label)
        image_outer_circle = draw_outer_circle(copy.copy(ori_img), label)

        if arg.watershed == True:
            watershed_label = np.expand_dims(label, 0).repeat(3, axis=0)
            watershed_label = np.transpose(watershed_label, [1, 2, 0])
            mask = watershed(watershed_label)
            mask = mask[:, :, 0]
            mask_color = label_draw_color(mask)
            image_mask_outer_circle = draw_outer_circle(copy.copy(ori_img), mask)
            # out = np.hstack((image, label, mask))
            # 原图，算法，算法+water，原图彩色，算法彩色，算法+water彩色，算法描边，算法+water描边
            out = np.hstack((channel_single2three(ori_img), channel_single2three(label), channel_single2three(mask),
                             img_color, label_color, channel_single2three(mask_color), image_outer_circle,
                             image_mask_outer_circle))
        else:
            # out = np.hstack((image, label))  # 同时可视化原图和测试结果图
            out = np.hstack((channel_single2three(ori_img), channel_single2three(label), img_color, label_color,
                             image_outer_circle))

        cv2.imwrite(os.path.join(save_dir, data.split('/')[-1]), out)

        fft = False
        if fft == True:
            img_float = np.float32(ori_img)
            dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_center = np.fft.fftshift(dft)
            crow, ccol = int(ori_img.shape[0] / 2), int(ori_img.shape[1] / 2)
            ma = np.zeros((ori_img.shape[0], ori_img.shape[1], 2), np.uint8)
            ma[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
            mask_img = dft_center * ma
            img_idf = np.fft.ifftshift(mask_img)
            img_idf = cv2.idft(img_idf)
            img = cv2.magnitude(img_idf[:, :, 0], img_idf[:, :, 1])
            os.makedirs(save_dir + '_color', exist_ok=True)
            label = cv2.resize(label, (w, h))
            save_img_mask(img, label, name=os.path.join(save_dir + '_color', data.split('/')[-1]))


if __name__ == '__main__':
    inference2()
