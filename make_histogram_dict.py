import cv2
import os
import json
from utils import hist_inform

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', type=str, required=True)

    return parser.parse_args()


def make_dict(base_img_folder):
    style_iter = os.listdir(base_img_folder)
    hist_dict = {}

    for style_img in style_iter:
        temp_dict = {}
        temp_style = cv2.imread(base_img_folder + '/' + style_img, cv2.IMREAD_COLOR)
        hist, num = hist_inform(temp_style)
        temp_dict['hist'] = hist.tolist()
        temp_dict['num'] = num
        hist_dict[style_img] = temp_dict

    with open('hist_features.json', 'w') as fp:
        json.dump(hist_dict, fp)

    print('finished!')

    return

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    make_dict(args.img_path)
