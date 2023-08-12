from PIL import Image
import torch
import math
from pytorch_msssim import ssim, ms_ssim
from torchvision import transforms


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_ssim(a, b):
    return ssim(a, b, data_range=1.).item()


def eval_psnr_ssim_msssim(ori_img_path, new_img_path):
    ori_img = Image.open(ori_img_path).convert('RGB')
    new_img = Image.open(new_img_path).convert('RGB')
    if ori_img.size != new_img.size:
        new_img = new_img.resize(ori_img.size)
    ori_x = transforms.ToTensor()(ori_img).unsqueeze(0)
    new_x = transforms.ToTensor()(new_img).unsqueeze(0)
    return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x), compute_msssim(ori_x, new_x)


def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result
