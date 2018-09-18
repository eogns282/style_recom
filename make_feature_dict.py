from keras.applications.vgg19 import VGG19
from keras.models import Model
import os
import json
from utils import load_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', type=str, required=True)

    return parser.parse_args()


def make_dict(base_img_folder):
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    img_list = os.listdir(base_img_folder)
    features_dict_base = {}

    for img_name in img_list:
        img_path = base_img_folder + '/' + img_name
        x = load_image(img_path)
        fc7_features = model.predict(x)
        features_dict_base[img_name] = fc7_features[0].tolist()

    with open('style_features.json', 'w') as fp:
        json.dump(features_dict_base, fp)

    print('finished!')

    return


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    make_dict(args.img_path)