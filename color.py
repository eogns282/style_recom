import cv2
import numpy as np
import json

def make_hist(img):
    unique, counts = np.unique(np.reshape(img, [-1, 3]), axis=0, return_counts=True)
    origin_hist = np.reshape(sorted(counts, reverse=True), [-1, 1])
    return origin_hist


def hist_inform(img):
    w, h = img.shape[0], img.shape[1]

    origin_hist = make_hist(img)
    img_num = origin_hist.shape[0]

    img_hist = (origin_hist).ravel().astype('float32')
    nor_img_hist = img_hist / (w * h * 0.001)

    return nor_img_hist, img_num


def load_hist_dict(json_file):
    json_data = open(json_file).read()
    hist_dict = json.loads(json_data)
    return hist_dict

	
def compare_fn(content_img, style_hist_dict):
    img1 = cv2.imread(content_img, cv2.IMREAD_COLOR)
    nor_img_hist1, img_num1 = hist_inform(img1)

    temp_dict = {}
    for style_img in style_hist_dict.keys():
        nor_img_hist2 = np.array(style_hist_dict[style_img]['hist']).ravel().astype('float32')
        img_num2 = style_hist_dict[style_img]['num']

        min_len = min(img_num1, img_num2)

        final_hist1 = nor_img_hist1[:min_len]
        final_hist2 = nor_img_hist2[:min_len]

        compare_val = cv2.compareHist(final_hist1, final_hist2, 1)
        temp_dict[style_img] = compare_val

    return temp_dict

def color_rank(content_img, style_hist_dict, reverse=False, top_n=5):
    temp = compare_fn(content_img, style_hist_dict)
    result = sorted(temp, key=temp.get, reverse=reverse)[:top_n]
    return result