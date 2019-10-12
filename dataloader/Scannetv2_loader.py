import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class Scannetv2(Dataset):

    def __init__(self, root_path, data_list):
        super().__init__()

        self.root_path = root_path
        self.data_list = data_list

        self.img_names = []
        self.left_imgs = []
        self.right_imgs = []
        self.disp_imgs = []

        self._load_data_path()

    def _load_data_path(self):
        scene_file = os.path.join(self.data_list)
        scene_list = np.loadtxt(scene_file, dtype='str', delimiter=',')

        data_path_loader = tqdm(scene_list, desc='Load data path', leave=True)
        for scene_name, scene_id in data_path_loader:

            left_path = os.path.join(
                    self.root_path, scene_name,
                    'left', f'{scene_id}.png')
            right_path = os.path.join(
                    self.root_path, scene_name,
                    'right', f'{scene_id}.png')
            disp_path = os.path.join(
                    self.root_path, scene_name,
                    'disp', f'{scene_id}_mesh_depth.tiff')
            self.img_names.append((scene_name, scene_id))
            self.left_imgs.append(left_path)
            self.right_imgs.append(right_path)
            self.disp_imgs.append(disp_path)
        self.len = len(self.left_imgs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = {}
        # TODO : Rewrite dataloader
        data['name'] = self.img_names[idx]

        # three frames
        data['frame1'] = Image.open(self.left_imgs[idx])
        data['frame2'] = Image.open(self.left_imgs[idx])
        data['frame3'] = Image.open(self.left_imgs[idx])
        # relative extrinsic parameters
        data['extr1'] = np.empty((4, 4))
        data['extr2'] = np.empty((4, 4))

        data['disp'] = Image.open(self.disp_imgs[idx])
        return data

    def customed_collate_fn(self, batch):
        # trans_height, trans_width = 240, 320
        tensor_transform = transforms.Compose([
            # transforms.Resize((trans_height, trans_width)),
            transforms.ToTensor(),
        ])

        def _transform_fn(key, value):
            if key == 'disp':
                value = tensor_transform(value).squeeze(0).type(torch.float32)
                value[value == float("Inf")] = 0
            else:
                value = tensor_transform(value).type(torch.float32)
            return value
        keys = list(batch[0].keys())
        values = {}
        for key in keys:
            if key == 'name':
                this_value = [one_batch[key] for one_batch in batch]
            else:
                this_value = \
                    torch.stack(
                        [_transform_fn(key, one_batch[key])
                            for one_batch in batch],
                        0,
                        out=None
                    )
            values[key] = this_value
        return values


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    root = "/tmp2/tsunghan/disparity_data/scannet_disparity_data/"
    data_list = "data_list/scannetv2"
    test_dataset = Scannetv2(root, data_list)
    test_loader = \
        DataLoader(
            test_dataset, batch_size=32,
            shuffle=False, num_workers=8
        )

    for batch in tqdm(test_loader):
        # print (batch)
        exit()
