import numpy as np
from skimage.measure import structural_similarity as ssim
import cv2

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_img', type=str, required=True)
    parser.add_argument('--result_img', type=str, required=True)

    return parser.parse_args()


def calc_ssim(content_img_path, result_img_path):
    content_img = cv2.imread(content_img_path)
    result_img = cv2.imread(result_img_path)

    content_img = cv2.resize(content_img, (np.shape(result_img)[1], np.shape(result_img)[0]))

    ssim_val = ssim(content_img, result_img, multichannel=True)

    return ssim_val


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    result = calc_ssim(args.content_img, args.result_img)
    print(result)

    return result


if __name__ == '__main__':
    main()
