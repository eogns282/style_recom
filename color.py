import cv2
import numpy as np
import json
from utils import hist_inform


def compare_fn(content_img, style_dict):
    img1 = cv2.imread(content_img, cv2.IMREAD_COLOR)
    nor_img_hist1, img_num1 = hist_inform(img1)

    temp_dict = {}
    for style_img in style_dict.keys():
        nor_img_hist2 = np.array(style_dict[style_img]['hist']).ravel().astype('float32')
        img_num2 = style_dict[style_img]['num']

        min_len = min(img_num1, img_num2)

        final_hist1 = nor_img_hist1[:min_len]
        final_hist2 = nor_img_hist2[:min_len]

        compare_val = cv2.compareHist(final_hist1, final_hist2, 1)
        temp_dict[style_img] = compare_val

    return temp_dict


def color_rank(content_img, style_dict, reverse=False, top_n=5):
    hist_dict_base = json.loads(open(style_dict).read())
    result_dict = compare_fn(content_img, hist_dict_base)
    result = sorted(result_dict, key=result_dict.get, reverse=reverse)[:top_n]

    return result
