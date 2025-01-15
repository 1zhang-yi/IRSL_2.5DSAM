import os
import nibabel as nib
import argparse
from tqdm import tqdm
from glob import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str,
                   default='Abdomen/Training', help='download path for Synapse data')
parser.add_argument('--dst_path', type=str,
                   default='npz_data', help='root dir for data')
parser.add_argument('--length', type=int, default=5)
parser.add_argument('--use_normalize', default=True, help='use normalize')
args = parser.parse_args()

train_data = [5, 6, 7, 9, 10, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 37, 39, 40]
test_data = [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]

hashmap = {1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 5, 7: 6, 8: 7, 9: 0, 10: 0, 11: 8, 12: 0, 13: 0}

class_map = {
    1: 'spleen',
    2: 'rightkidney',
    3: 'leftkidney',
    4: 'gallbladder',
    5: 'liver',
    6: 'stomach',
    7: 'aorta',
    8: 'pancreas'
}

case_slice = {
    "0001": [61, 146],
    "0002": [67, 138],
    "0003": [97, 197],
    "0004": [71, 139],
    "0005": [23, 116],
    "0006": [67, 130],
    "0007": [68, 162],
    "0008": [76, 147],
    "0009": [19, 148],
    "0010": [62, 147],
    "0021": [75, 142],
    "0022": [45, 88],
    "0023": [45, 95],
    "0024": [62, 123],
    "0025": [36, 84],
    "0026": [46, 115],
    "0027": [39, 87],
    "0028": [36, 88],
    "0029": [18, 99],
    "0030": [79, 152],
    "0031": [6, 92],
    "0032": [71, 143],
    "0033": [49, 103],
    "0034": [47, 97],
    "0035": [48, 93],
    "0036": [89, 183],
    "0037": [54, 98],
    "0038": [42, 99],
    "0039": [46, 89],
    "0040": [68, 169]
}


def preprocess_train_data(image_list: list, mode: str) -> None:
    os.makedirs(f"{args.dst_path}/{mode}", exist_ok=True)
    a_min, a_max = -125, 275
    pbar = tqdm(image_list, total=len(image_list))
    for image_file in pbar:
        label_file = image_file.replace('img', 'label')
        case_number = image_file.split('/')[-1][3:7]

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()
        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 1, 0))  # [D, W, H]
        label_data = np.transpose(label_data, (2, 1, 0))


        counter = 1
        for k in sorted(hashmap.keys()):
            assert counter == k
            counter += 1
            label_data[label_data == k] = hashmap[k]

        width = case_slice[case_number]
        start_slice = width[0]
        end_slice = width[1]


        left = args.length // 2
        right = args.length - left
        for i in range(start_slice, end_slice + 1):
            start_id = i - left
            end_id = i + right
            if end_id >= end_slice + 1:
                label_part = label_data[end_slice + 1 - args.length: end_slice + 1, :, :]
                img_part = image_data[end_slice + 1 - args.length: end_slice + 1, :, :]
                idx = end_slice + 1 - args.length
            else:
                label_part = label_data[start_id: end_id, :, :]
                img_part = image_data[start_id: end_id, :, :]
                idx = start_id

            if label_part.shape[0] == img_part.shape[0] and img_part.shape[0] == args.length:
                for class_id in np.unique(label_part[2]):
                    if class_id != 0:
                        label_part_id = (label_part == class_id)
                        class_id = int(class_id)
                        organ = class_map[class_id]
                        path = f'{args.dst_path}/{mode}/{case_number}_{idx}_{organ}{class_id}.npz'
                        np.savez(path, image=img_part, label=label_part_id, start_slice=idx, organ=organ)
    pbar.close()


def preprocess_test_data(image_list: list, mode: str) -> None:
    os.makedirs(f"{args.dst_path}/{mode}", exist_ok=True)
    a_min, a_max = -125, 275
    pbar = tqdm(image_list, total=len(image_list))
    for image_file in pbar:
        label_file = image_file.replace('img', 'label')
        case_number = image_file.split('/')[-1][3:7]

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()
        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 1, 0))  # [D, W, H]
        label_data = np.transpose(label_data, (2, 1, 0))


        counter = 1
        for k in sorted(hashmap.keys()):
            assert counter == k
            counter += 1
            label_data[label_data == k] = hashmap[k]

        width = case_slice[case_number]
        start_slice = width[0]
        end_slice = width[1]

        left = args.length // 2
        right = args.length - left

        for i in range(start_slice, end_slice+1):
            label_part = label_data[i, :, :]
            idx = i
            for class_id in np.unique(label_part):
                if class_id != 0:
                    label_ = label_data[i-left: i+right, :, :]
                    image_ = image_data[i-left: i+right, :, :]
                    if label_.shape[0] == args.length:
                        label_part_id = (label_ == class_id)
                        class_id = int(class_id)
                        organ = class_map[class_id]
                        path = f'{args.dst_path}/{mode}/{case_number}_{idx}_{organ}{class_id}.npz'
                        np.savez(path, image=image_, label=label_part_id, start_slice=idx, organ=organ)
    pbar.close()

if __name__ == "__main__":
    data_root = f"{args.src_path}/img"
    label_root = f"{args.src_path}/label"

    image_files = sorted(glob(f"{data_root}/*.nii.gz"))
    train_image = []
    test_image = []
    for path in image_files:
        number = path.split('/')[-1][3: 7]
        number = int(number)
        if number in train_data:
            train_image.append(path)
        elif number in test_data:
            test_image.append(path)
        else:
            print("Error number {} !!!!!!".format(number))

    # print("Preprocess training data........")
    # preprocess_train_data(train_image, f'train_{args.length}')
    print("Preprocess testing data........")
    preprocess_test_data(test_image, f'test_{args.length}')



