import numpy as np
import torch
import nibabel as nib
from glob import glob
from torch.utils.data import Dataset
from utils import init_point_sampling

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





class Synapse_train_dataset(Dataset):
    def __init__(self, base_dir, split, length, point_num=1, transform=None):
        self.transform = transform  # using transform in torch!
        self.data_dir = f'{base_dir}/{split}' #npz_data/train(test)
        self.length = length
        self.point_num = point_num
        self.data = sorted(glob(f"{self.data_dir}/*.npz"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        data = np.load(data_path)
        image = data['image']
        label = data['label']
        assert image.shape[0] == label.shape[0] == self.length
        start_slice = data['start_slice'].tolist()
        organ = data['organ'].tolist()
        case_number = data_path.split('/')[-1][:4]

        image_copy = np.zeros((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32)
        image_copy[1:, :, :] = image[0:image.shape[0] - 1, :, :]
        image_res = image - image_copy
        image_res[0, :, :] = 0
        image_res = np.abs(image_res)
        image_res = torch.from_numpy(image_res.astype(np.float32)).unsqueeze(1)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(1)
        image = image.repeat(1, 3, 1, 1)
        label = torch.from_numpy(label.astype(np.float32)).to(torch.int64)
        point_list = []
        label_list = []
        for i in range(label.shape[0]):
            label_slice = label[i]
            point_coords, point_label = init_point_sampling(label_slice, self.point_num)
            point_list.append(point_coords)
            label_list.append(point_label)

        point_coords_stack = torch.stack(point_list, dim=0)
        point_label_stack = torch.stack(label_list, dim=0)


        sample = {'image': image,
                  'image_res': image_res,
                  'label': label.unsqueeze(1),
                  'point_coords': point_coords_stack,
                  'point_labels': point_label_stack,
                  'organ': organ,
                  'case_number': case_number,
                  'start_slice': start_slice
                  }
        return sample

class Synapse_test_dataset(Dataset):
    def __init__(self, base_dir, split, point_num=1, transform=None):
        self.transform = transform  # using transform in torch!
        self.data_dir = f'{base_dir}/{split}' #npz_data/train(test)
        self.point_num = point_num
        self.data = sorted(glob(f"{self.data_dir}/*.npz"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        data = np.load(data_path)
        image = data['image']
        label = data['label']

        start_slice = data['start_slice'].tolist()
        organ = data['organ'].tolist()
        case_number = data_path.split('/')[-1][:4]

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(1)
        image = image.repeat(1, 3, 1, 1)
        label = torch.from_numpy(label.astype(np.float32)).to(torch.int64)
        point_list = []
        label_list = []
        for i in range(label.shape[0]):
            label_slice = label[i]
            point_coords, point_label = init_point_sampling(label_slice, self.point_num)
            point_list.append(point_coords)
            label_list.append(point_label)

        point_coords_stack = torch.stack(point_list, dim=0)
        point_label_stack = torch.stack(label_list, dim=0)


        sample = {'image': image,
                  'label': label.unsqueeze(1),
                  'point_coords': point_coords_stack,
                  'point_labels': point_label_stack,
                  'organ': organ,
                  'case_number': case_number,
                  'start_slice': start_slice
                  }
        return sample