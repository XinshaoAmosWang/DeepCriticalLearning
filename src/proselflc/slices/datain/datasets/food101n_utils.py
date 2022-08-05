from __future__ import print_function

import os

import numpy as np

DATA_ROOT = (
    "/home/xinshao/tpami_proselflc_experiments_calibration/" "input_dir/Food101-N"
)
TRAIN_PATH = DATA_ROOT + "/Food-101N_release"
TEST_PATH = DATA_ROOT + "/food-101"


def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def gen_train_list():
    root_data_path = TRAIN_PATH + "/meta/imagelist.tsv"
    class_list_path = TRAIN_PATH + "/meta/classes.txt"
    file_path_prefix = TRAIN_PATH + "/images"

    map_name2cat = dict()
    with open(class_list_path) as fp:
        fp.readline()  # skip first line

        for i, line in enumerate(fp):
            row = line.strip()
            map_name2cat[row] = i
    num_class = len(map_name2cat)
    print(map_name2cat)
    print("Num Classes: ", num_class)

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        fp.readline()  # skip first line

        for line in fp:
            row = line.strip().split("/")
            class_name = row[0]
            targets.append(map_name2cat[class_name])
            img_list.append(os.path.join(file_path_prefix, line.strip()))

    targets = np.array(targets)
    img_list = np.array(img_list)
    assert len(img_list) == len(targets)
    print(targets[-1])
    print(img_list[-1])
    print("Num Train Images: ", len(img_list))
    print()

    save_dir = check_folder(DATA_ROOT + "/image_list")
    np.save(os.path.join(save_dir, "train_images"), img_list)
    np.save(os.path.join(save_dir, "train_targets"), targets)

    return map_name2cat


def gen_test_list(arg_map_name2cat):
    map_name2cat = arg_map_name2cat
    root_data_path = TEST_PATH + "/meta/test.txt"

    file_path_prefix = TEST_PATH + "/images"

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        for line in fp:
            row = line.strip().split("/")
            class_name = row[0]
            targets.append(map_name2cat[class_name])
            img_list.append(os.path.join(file_path_prefix, line.strip() + ".jpg"))

    targets = np.array(targets)
    img_list = np.array(img_list)
    assert len(img_list) == len(targets)
    print(targets[-1])
    print(img_list[-1])
    print("Num Test Images: ", len(img_list))
    print()

    save_dir = check_folder(DATA_ROOT + "/image_list")
    np.save(os.path.join(save_dir, "test_images"), img_list)
    np.save(os.path.join(save_dir, "test_targets"), targets)


if __name__ == "__main__":
    map_name2cat = gen_train_list()
    gen_test_list(map_name2cat)
