from PIL import Image
import argparse
import cv2
import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from models.PSMnet import PSMNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader.Scannetv2_loader import Scannetv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PSMNet inference')
parser.add_argument('--maxdisp', type=int, default=192, help='max diparity')
parser.add_argument('--model-path', default=None, help='path to the model')
parser.add_argument('--datadir', default="/work/kaikai4n/resized_stereo/"
                    , help='data directory')
args = parser.parse_args()


#mean = [0.406, 0.456, 0.485]
#std = [0.225, 0.224, 0.229]
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))


def main():

    model = PSMNet(args.maxdisp).to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    state = torch.load(args.model_path)
    if len(device_ids) == 1:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            namekey = k[7:] # remove `module.`
            new_state_dict[namekey] = v
        state['state_dict'] = new_state_dict

    model.load_state_dict(state['state_dict'])
    print('load model from {}'.format(args.model_path))
    print('epoch: {}'.format(state['epoch']))
    print('3px-error: {}%'.format(state['error']))

    model.eval()
 
    val_dataset =  Scannetv2(args.datadir, "dataloader/data_list/scannetv2", 'val')
    validate_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, collate_fn=val_dataset.customed_collate_fn)

    N = 0
    for batch in tqdm(validate_loader):
        
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        with torch.no_grad():
            _, _, disp = model(left_img, right_img)

        disp = disp.squeeze(0).detach().cpu().numpy()
        disp_img = Image.fromarray(disp).convert('F')
        save_path = f'disparity_{N}.tiff'
        disp_img.save(os.path.join("raw_vis", save_path))
        #plt.figure(figsize=(8.84, 3.84))
        #plt.axis('off')
        #plt.imshow(disp)
        #plt.colorbar()
        #save_path = f'disparity_{N}.png'
        #plt.savefig(os.path.join("validation", save_path), dpi=100)
        #plt.close()
        N += 1
        if N > 10:
            exit()
        # print('save diparity map in {}'.format(args.save_path))

if __name__ == '__main__':
    main()
