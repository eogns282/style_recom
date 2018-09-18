import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--style_img', type=str, required=True)
    parser.add_argument('--result_img', type=str, required=True)
    parser.add_argument('--seed', type=int, default=7777)

    return parser.parse_args()


def load_img(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


def rand_unit_vec(size, seed=None):
    if seed != None:
        np.random.seed(seed)
    temp = np.random.randn(size)
    norm_temp = temp / np.sqrt(np.sum(np.square(temp)))

    return norm_temp


def normal_kl(mu_1, std_1, mu_2, std_2):
    # mu_1, std_1 is style feature
    # mu_2, std_2 is result feature
    return (np.log(std_2 / std_1)) + ((std_1 ** 2 + ((mu_1 - mu_2) ** 2)) / (2 * (std_2 ** 2))) - (0.5)


def normal_dist(img_path, rand_vec, feature_extractor):
    img = load_img(img_path)
    feature_map = feature_extractor([img])

    result_0 = np.reshape(np.dot(feature_map[0], rand_vec[:64]), -1).tolist()
    result_1 = np.reshape(np.dot(feature_map[1], rand_vec[64:192]), -1).tolist()
    result_2 = np.reshape(np.dot(feature_map[2], rand_vec[192:448]), -1).tolist()
    result_3 = np.reshape(np.dot(feature_map[3], rand_vec[448:960]), -1).tolist()
    result_4 = np.reshape(np.dot(feature_map[4], rand_vec[960:]), -1).tolist()

    scalar_data = result_0 + result_1 + result_2 + result_3 + result_4

    mu = np.mean(scalar_data)
    std = np.std(scalar_data)

    return mu, std


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    base_model = VGG19(weights='imagenet')
    feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    get_feature = K.function([base_model.layers[0].input],
                             [base_model.get_layer(name).output[0] for name in feature_layers])

    np.random.seed(args.seed)
    vec_array = np.array([rand_unit_vec(1472) for _ in range(0, 128)])

    kl_arr = []
    for rand_vec_idx in range(128):
        mu_1, std_1 = normal_dist(img_path=args.style_img, rand_vec=vec_array[rand_vec_idx],
                                  feature_extractor=get_feature)
        mu_2, std_2 = normal_dist(img_path=args.result_img, rand_vec=vec_array[rand_vec_idx],
                                  feature_extractor=get_feature)
        kl_div = normal_kl(mu_1, std_1, mu_2, std_2)
        kl_arr.append(kl_div)

    result = np.mean(kl_arr)
    print(result)

    return result


if __name__ == '__main__':
    main()
