from structure import Recomm
from color import load_hist_dict, color_rank
import shutil

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--top_n', type=int, default=5)

    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    print('json file loading...')
    feature_arr = Recomm(content_image=args.content, style_dict='./json_data/gray_style_features.json')
    hist_dict = load_hist_dict('./json_data/hist_features.json')

    ref_arr = color_rank(args.content, hist_dict, top_n=540)
    ans = {}
    for style_img in feature_arr:
        ans[style_img] = ref_arr.index(style_img)

    result_arr = sorted(ans, key=ans.get)[:args.top_n]

    k = 1
    for style_img in result_arr:
        shutil.copy('./base_style/' + style_img, './result/' + str(k) + '.jpg')
        k += 1

    print('Finished! Recommended images are in the "result" folder.')

    return


if __name__ == '__main__':
    main()