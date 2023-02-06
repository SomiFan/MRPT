"""
explore_grad.py 2022/7/6 10:41
Written by Wensheng Fan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import numpy as np
from skimage import io
from utils import sobelfilter2d, normalize_torch
if __name__ == "__main__":

    imdata_fused = scipy.io.loadmat(r"F:\ResearchData\dataset\WV3\forpresentation\rr\rr10.mat")
    i_f = imdata_fused['oms_patch'].astype(np.float32).transpose((2, 0, 1))
    i_f = torch.from_numpy(i_f).float().unsqueeze(0).cuda()
    i_f_gray = torch.mean(i_f, 1, True)
    i_f_sfed = torch.abs(sobelfilter2d(i_f_gray))

    opan = imdata_fused['lpan_patch'].astype(np.float32)
    #opan = io.imread(r"F:\ResearchData\dataset\QB\forpresentation\opan\opan_patch12.tif").astype(np.float32)
    opan = torch.from_numpy(opan).float().unsqueeze(0).unsqueeze(0).cuda()
    opan_sfed = torch.abs(sobelfilter2d(opan))

    """
    res_sfed = torch.abs(i_f_sfed-opan_sfed)
    res_sfed_x = normalize_torch(res_sfed[0, 0, :, :]) * 255
    io.imsave(f'./res_sfed_x.jpg', res_sfed_x.cpu().data.numpy().astype(np.uint8))
    res_sfed_y = normalize_torch(res_sfed[0, 1, :, :]) * 255
    io.imsave(f'./res_sfed_y.jpg', res_sfed_y.cpu().data.numpy().astype(np.uint8))
    res_sfed = normalize_torch(torch.sum(res_sfed, dim=1)) * 255
    res_sfed = res_sfed.cpu().data.numpy()[0]
    io.imsave(f'./res_sfed.jpg', res_sfed.astype(np.uint8))
    """
    i_f_sfed_x = normalize_torch(i_f_sfed[0, 0, :, :]) * 255
    io.imsave(f'D:\Document\Papers\showmetheresults\sapt/ms_sfed_x.jpg', i_f_sfed_x.cpu().data.numpy().astype(np.uint8))
    i_f_sfed_y = normalize_torch(i_f_sfed[0, 1, :, :]) * 255
    io.imsave(f'D:\Document\Papers\showmetheresults\sapt/ms_sfed_y.jpg', i_f_sfed_y.cpu().data.numpy().astype(np.uint8))
    i_f_sfed = normalize_torch(torch.sum(i_f_sfed, dim=1)) * 255
    i_f_sfed = i_f_sfed.cpu().data.numpy()[0]
    io.imsave(f'D:\Document\Papers\showmetheresults\sapt/ms_sfed.jpg', i_f_sfed.astype(np.uint8))
    opan_sfed_x = normalize_torch(opan_sfed[0, 0, :, :]) * 255
    io.imsave(f'D:\Document\Papers\showmetheresults\sapt/opan_sfed_x.jpg', opan_sfed_x.cpu().data.numpy().astype(np.uint8))
    opan_sfed_y = normalize_torch(opan_sfed[0, 1, :, :]) * 255
    io.imsave(f'D:\Document\Papers\showmetheresults\sapt/opan_sfed_y.jpg', opan_sfed_y.cpu().data.numpy().astype(np.uint8))
    opan_sfed = normalize_torch(torch.sum(opan_sfed, dim=1)) * 255
    opan_sfed = opan_sfed.cpu().data.numpy()[0]
    io.imsave(f'D:\Document\Papers\showmetheresults\sapt/opan_sfed.jpg', opan_sfed.astype(np.uint8))

    """
    imdata_fused = scipy.io.loadmat(r"D:\Document\experiments\DL_Pansharp\PNN_FT_QB\PNN_FT_QB_FR_QB12_237.mat")
    i_f = imdata_fused['I_F'].astype(np.float32).transpose((2, 0, 1))
    i_f = torch.from_numpy(i_f).float().unsqueeze(0).cuda()
    i_f_gray = torch.mean(i_f, 1, True)
    i_f_sfed = torch.abs(sobelfilter2d(i_f_gray))

    opan = io.imread(r"F:\ResearchData\dataset\QB\forpresentation\opan\opan_patch12.tif").astype(np.float32)
    opan = torch.from_numpy(opan).float().unsqueeze(0).unsqueeze(0).cuda()
    opan_sfed = torch.abs(sobelfilter2d(opan))

    res_sfed = torch.abs(i_f_sfed - opan_sfed)
    res_sfed_x = normalize_torch(res_sfed[0, 0, :, :]) * 255
    io.imsave(f'./res_sfed_x237.jpg', res_sfed_x.cpu().data.numpy().astype(np.uint8))
    res_sfed_y = normalize_torch(res_sfed[0, 1, :, :]) * 255
    io.imsave(f'./res_sfed_y237.jpg', res_sfed_y.cpu().data.numpy().astype(np.uint8))
    res_sfed = torch.sum(res_sfed, dim=1)
    res_sfed = res_sfed.cpu().data.numpy()[0]
    io.imsave(f'./res_sfed237.jpg', res_sfed.astype(np.uint8))
    i_f_sfed_x = normalize_torch(i_f_sfed[0, 0, :, :]) * 255
    io.imsave(f'./i_f_sfed_x237.jpg', i_f_sfed_x.cpu().data.numpy().astype(np.uint8))
    i_f_sfed_y = normalize_torch(i_f_sfed[0, 1, :, :]) * 255
    io.imsave(f'./i_f_sfed_y237.jpg', i_f_sfed_y.cpu().data.numpy().astype(np.uint8))
    i_f_sfed = normalize_torch(torch.sum(i_f_sfed, dim=1)) * 255
    i_f_sfed = i_f_sfed.cpu().data.numpy()[0]
    io.imsave(f'./i_f_sfed237.jpg', i_f_sfed.astype(np.uint8))
    """
    """
    imdata_fused = scipy.io.loadmat(r"D:\Document\experiments\DL_Pansharp\PNN_FT_QB\PNN_FT_QB_FR_QB12_237.mat")
    i_f = imdata_fused['I_F'].astype(np.float32).transpose((2, 0, 1))
    i_f = torch.from_numpy(i_f).float().unsqueeze(0).cuda()
    i_f_gray = torch.mean(i_f, 1, True)
    i_f_sfed = torch.abs(sobelfilter2d(i_f_gray))

    opan = io.imread(r"F:\ResearchData\dataset\WV3\forpresentation\fr\opan_patch12.tif").astype(np.float32)
    opan = torch.from_numpy(opan).float().unsqueeze(0).unsqueeze(0).cuda()
    opan_sfed = torch.abs(sobelfilter2d(opan))

    res_sfed = torch.abs(i_f_sfed - opan_sfed)
    res_sfed = normalize_torch(torch.sum(res_sfed, dim=1)) * 255
    res_sfed = res_sfed.cpu().data.numpy()[0]
    mask_grad = res_sfed
    mask_grad[res_sfed < 20] = 0
    mask_grad[res_sfed >= 20] = 255
    io.imsave(f'./mask_grad_qb12.jpg', mask_grad.astype(np.uint8))
    i_f = io.imread(r"D:\Document\experiments\DL_Pansharp\PNN_FT_QB\PNN_FT_QB_FR_QB12_237.jpg").astype(
        np.uint8).transpose((2, 0, 1))
    i_f = torch.from_numpy(i_f).unsqueeze(0)
    for i in range(3):
        i_f[0, i, :, :, ][res_sfed < 20] = 0
    io.imsave("./i_f_grad_masked_qb12.jpg", i_f.squeeze(0).numpy().transpose((1, 2, 0)))
    """