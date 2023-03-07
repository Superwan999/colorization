import argparse
import os
import collections

import numpy as np
import math
from scipy.ndimage import gaussian_filter
from PIL import Image
from statistics import mean


# Method 1. SSIM
def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)
    """
    # variables are initialized as suggested in the paper
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5

    # means
    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    # variances and covariances
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # normalize by unbiased estimate of std dev
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # eq. 4 of the paper
    vx = (uxx - ux * ux) * unbiased_norm
    vy = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # compute SSIM (eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim/D
    mssim = SSIM.mean()

    return mssim


def getSSIM(X, Y):
    """
       Computes the mean structural similarity between two images.
    """
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"
    nch = 1 if X.ndim == 2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[..., ch].astype(np.float64), Y[..., ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)


# Method 2. PSNR
def getPSNR(X, Y):
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    if rmse == 0:
        return 100
    else:
        return 20 * math.log10(255. / rmse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--path',
                        type=str,
                        default='D:/my_doc/inter/interview/neon/eval_images',
                        # default='D:/my_doc/inter/interview/neon/eval_images/968.jpg',
                        help='data path: image folder path for evaluation')
    args = parser.parse_args()

    # get image name(s)
    image_path = args.path
    if os.path.isdir(image_path):
        image_names = [os.path.join(image_path, image_id) for image_id in os.listdir(image_path)]
    else:
        image_names = [image_path]

    # get measurements
    ms = collections.defaultdict(dict)
    for image_name in image_names:
        image = Image.open(image_name)
        image_256 = image.resize((256, 256))

        # evaluation
        psnr = getPSNR(image_256, image_256)
        ssim = getSSIM(image_256, image_256)
        ms['spnr'][image_name] = psnr
        ms['ssim'][image_name] = ssim

    mean_spnr = mean(list(ms['spnr'].values()))
    mean_ssim = mean(list(ms['ssim'].values()))








