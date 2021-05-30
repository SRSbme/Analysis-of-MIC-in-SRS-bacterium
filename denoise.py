from bm3d.bm3d import run_bm3d, con_bm3d
from utils.nlm import run_nlm, con_nlm
from utils.gaussian import run_gaussian, con_gaussian
from utils.rof import run_rof, con_rof
from utils.unet import run_unet, con_unet, save_unet_pic


# 三个指标理论上, psnr越大越好,  rmse越小越好, cc越大越好.

# 因为自然噪声不是人为加上的，所以噪声类型不知道也不会是统一的噪声类型，
# 所以不可能一种算法适应所有数据，只能说某种算法针对该数据集综合效果最好
if __name__ == '__main__':
    # test
    type = 'rof'  # [bm3d, nlm, gaussian, rof]
    if type == 'bm3d':
        # im_dir为待去噪图片文件夹, sigma为可选超参数，sigma_list = [2, 5, 10, 20, 30, 40, 60, 80, 100]
        run_bm3d(im_dir='noise_datasets/20210128', sigma=2)

    elif type == 'nlm':
        run_nlm(im_dir='noise_datasets/20210128')

    elif type == 'gaussian':
        run_gaussian(im_dir='noise_datasets/20210128')

    elif type == 'rof':
        # Rudin - Osher - Fatemi
        run_rof(im_dir='noise_datasets/20210128')

    elif type == 'unet':
        run_unet(im_dir='noise_datasets/20210128')

    else:
        pass



# # 下面五种更科学, 因为指标是原始无噪声和经过去噪算法产生的结果计算得来的
# if __name__ == '__main__':
#
#     type = 'bm3d'  # [bm3d, nlm, gaussian, rof, unet]
#     if type == 'bm3d':
#         # im_dir为待去噪图片文件夹, sigma为可选超参数，sigma_list = [2, 5, 10, 20, 30, 40, 60, 80, 100]
#         con_bm3d(original_img_dir='datasets/original_img', noise_im_dir='datasets/noise_im', sigma=5)
#
#     elif type == 'nlm':
#         con_nlm(original_img_dir='datasets/original_img', noise_im_dir='datasets/noise_im')
#
#     elif type == 'gaussian':
#         con_gaussian(original_img_dir='datasets/original_img', noise_im_dir='datasets/noise_im')
#
#     elif type == 'rof':
#         # Rudin - Osher - Fatemi
#         con_rof(original_img_dir='datasets/original_img', noise_im_dir='datasets/noise_im')
#
#     elif type == 'unet':
#         # 训练阶段，更新权重，无法保存最优去噪图
#         # con_unet()
#         # 保存最优去噪图
#         save_unet_pic(original_img_dir='datasets/original_img', noise_im_dir='datasets/noise_im')
#
#     else:
#         pass

