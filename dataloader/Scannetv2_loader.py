import os, sys
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np

class Scannetv2(Dataset):

    def __init__(self, root_path, data_list, mode):
        super().__init__()
        if mode not in {'train', 'test', 'val'}:
            raise ValueError('Only support train, test and val mode.')

        self.mode = mode
        self.root_path = root_path
        self.data_list = data_list

        self.left_imgs = []
        self.right_imgs = []
        self.disp_imgs = []

        self._load_data_path()

    def _load_data_path(self):
        # TODO : split scene name list for training, testing, validation
        scene_file = os.path.join(self.data_list, f"{self.mode}_list.txt")
        scene_list = np.loadtxt(scene_file,dtype='str')

        for scene_name in tqdm(scene_list, desc='Load data path', leave=True):
            img_file = os.path.join(self.data_list, scene_name, "image_name_list.txt")
            img_list = np.loadtxt(img_file, dtype='str')

            left_path = [os.path.join(self.root_path, scene_name, 'left', f'{x}.png') for x in img_list]
            right_path = [os.path.join(self.root_path, scene_name, 'right', f'{x}.png') for x in img_list]
           
            self.left_imgs += left_path
            self.right_imgs += right_path
         
            if self.mode != 'test':
                disp_path = [os.path.join(self.root_path, scene_name, 'mesh_images', 
                                                f'{x}_mesh_depth.png') for x in img_list]
                self.disp_imgs += disp_path
            
        self.len = len(self.left_imgs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = {}
        data['left'] = Image.open(self.left_imgs[idx])
        data['right'] = Image.open(self.right_imgs[idx])
        if self.mode != 'test':
            data['disp'] = Image.open(self.disp_imgs[idx])
        return data

    def customed_collate_fn(self, batch):
        trans_height, trans_width = 480, 640
        tensor_transform = transforms.Compose([
            transforms.Resize((trans_height, trans_width)),
            transforms.ToTensor(),
        ])
        def _transform_fn(key, value):
            if key == 'disp':
                value = tensor_transform(value).squeeze(0).type(torch.float32)
                value /= 4000.00
            else:
                value = tensor_transform(value)
                #value = value.permute(2, 1, 0)
            return value
        keys = list(batch[0].keys())
        values = {}
        for key in keys:
            this_value = torch.stack([_transform_fn(key, one_batch[key]) for one_batch in batch], 0, out=None)
            values[key] = this_value
        return values

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    root = "/tmp2/tsunghan/disparity_data/scannet_disparity_data/"
    data_list = "data_list/scannetv2"
    test_dataset = Scannetv2(root, data_list)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    for batch in tqdm(test_loader):
        #print (batch)
        exit()

    

