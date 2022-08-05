# Copy From https://github.com/pxiangwu/PLC/
# blob/8c2bcfd25e538fdcdb637a820ac340e9ee084e59/
# clothing1m/data_clothing1m.py#L9


def get_train_labels(
    data_root="",
):
    train_file_list = data_root + "/annotations/noisy_train_key_list.txt"
    noise_label_file = data_root + "/annotations/noisy_label_kv.txt"

    # read train images
    fid = open(train_file_list)
    train_list = [line.strip() for line in fid.readlines()]
    fid.close()

    fid = open(noise_label_file)
    label_list = [line.strip().split(" ") for line in fid.readlines()]

    label_map = dict()
    for m in label_list:
        label_map[m[0]] = m[1]

    train_labels = []
    for t in train_list:
        label = label_map[t]
        train_labels.append(label)

    with open(data_root + "/annotations/my_train_label.txt", "w") as fid:
        for p in train_labels:
            fid.write("{}\n".format(p))

    return train_labels


def get_val_test_labels(
    data_root="",
):
    val_file_list = data_root + "/annotations/clean_val_key_list.txt"
    test_file_list = data_root + "/annotations/clean_test_key_list.txt"
    clean_label_file = data_root + "/annotations/clean_label_kv.txt"

    # read val images
    fid = open(val_file_list)
    val_list = [line.strip() for line in fid.readlines()]
    fid.close()

    # read test images
    fid = open(test_file_list)
    test_list = [line.strip() for line in fid.readlines()]
    fid.close()

    fid = open(clean_label_file)
    label_list = [line.strip().split(" ") for line in fid.readlines()]
    fid.close()

    label_map = dict()
    for m in label_list:
        label_map[m[0]] = m[1]

    val_labels = []
    for t in val_list:
        label = label_map[t]
        val_labels.append(label)

    test_labels = []
    for t in test_list:
        label = label_map[t]
        test_labels.append(label)

    with open(data_root + "/annotations/my_val_label.txt", "w") as fid:
        for p in val_labels:
            fid.write("{}\n".format(p))

    with open(data_root + "/annotations/my_test_label.txt", "w") as fid:
        for p in test_labels:
            fid.write("{}\n".format(p))


if __name__ == "__main__":
    data_root = "/home/xinshao/tpami_proselflc_experiments/input_dir/clothing1m"

    # may be noisy
    get_train_labels(
        data_root=data_root,
    )

    # clean labels
    get_val_test_labels(
        data_root=data_root,
    )
