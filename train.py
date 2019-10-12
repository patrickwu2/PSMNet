import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from args import get_args
from models.PSMnet import PSMNet
from models.smoothloss import SmoothL1Loss
from dataloader.Scannetv2_loader import Scannetv2

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

args = get_args()

# safely create saved model, log path before training
model_dir = os.path.join(args.experiment_dir, 'model')
log_dir = os.path.join(args.experiment_dir, 'log')
img_dir = os.path.join(args.experiment_dir, 'image')

os.makedirs(args.experiment_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

log_file_name = f'PSMNet_b{args.batch_size}_lr{args.lr}.log'
log_f = open(os.path.join(log_dir, log_file_name), 'w')

# imagenet
mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [3]

device = torch.device('cuda')


def main(args):

    train_dataset =  \
        Scannetv2(
            args.datadir,
            "dataloader/data_list/scannet_train_list.csv"
        )
    train_loader = \
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_dataset.customed_collate_fn
        )
    val_dataset = \
        Scannetv2(
            args.datadir,
            "dataloader/data_list/scannet_test_list.csv"
        )
    validate_loader = \
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=val_dataset.customed_collate_fn
        )

    step = 0
    best_error = 100.0

    model = PSMNet(args.maxdisp).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    criterion = SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.model_path is not None:
        state = torch.load(args.model_path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        best_error = state['error']
        print('load model from {}'.format(args.model_path))

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ))

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        step = train(model, train_loader, optimizer, criterion, step)
        adjust_lr(optimizer, epoch)

        if epoch % args.save_per_epoch == 0:
            model.eval()
            error = validate(model, validate_loader, epoch)
            best_error = save(model, optimizer, epoch, step, error, best_error)


def validate(model, validate_loader, epoch):
    '''
    validate 40 image pairs
    '''
    num_batches = len(validate_loader)
    idx = np.random.randint(num_batches)

    avg_error = 0.0
    i = 0
    for batch in tqdm(validate_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = (target_disp > 0)
        mask = mask.detach_()

        with torch.no_grad():
            _, _, disp = model(left_img, right_img)

        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat = (
                    ((delta >= 3.0)
                    + (delta >= 0.05 * (target_disp[mask]))) == 2
                )
        error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100

        avg_error += error
        if i == idx:
            left_save = left_img
            disp_save = disp
            save_image(left_save[0], disp_save[0], epoch)

        i += 1
    avg_error = avg_error / num_batches
    # write log to stdout, log file
    print('epoch: {:03} | 3px-error: {:.5}%'.format(
        epoch, avg_error
    ))
    print('epoch: {:03} | 3px-error: {:.5}%'.format(
        epoch, avg_error
    ), file=log_f)

    return avg_error


def save_image(left_image, disp, epoch):

    saved_name = f'epoch_{epoch}'

    # save color (left) image
    color = left_image.detach().cpu().numpy()
    color = np.transpose(color, (1, 2, 0)) * 255
    color_img = Image.fromarray(color.astype(np.uint8))

    color_img.save(os.path.join(img_dir, f'left_{saved_name}.png'))

    # save disparity image
    disp_img = disp.detach().cpu().numpy()
    plt.axis('off')  # hide axis
    plt.imshow(disp_img, cmap='jet')
    plt.colorbar()

    plt.savefig(os.path.join(img_dir, f'disp_{saved_name}.png'))


def train(model, train_loader, optimizer, criterion, step):
    '''
    train one epoch
    '''
    for batch in tqdm(train_loader):
        step += 1
        optimizer.zero_grad()
        # read batch
        frame1 = batch['frame1'].to(device)
        frame2 = batch['frame2'].to(device)
        frame3 = batch['frame3'].to(device)
        extr1 = batch['extr1'].to(device)
        extr2 = batch['extr2'].to(device)
        target_disp = batch['disp'].to(device)
        mask = (target_disp > 0)
        mask = mask.detach_()
        # feed into model
        disp1, disp2, disp3 = model(frame1, frame2, frame3, extr1, extr2)
        print(disp1.size)
        exit()
        loss1, loss2, loss3 = \
            criterion(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
        total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3

        total_loss.backward()
        optimizer.step()

        # print(step)

        if step % args.log_per_step == 0:
            print('step: {:05} | total loss: {:.5} \
                    | loss1: {:.5} | loss2: {:.5} | \
                    loss3: {:.5}'.format(
                        step, total_loss.item(),
                        loss1.item(), loss2.item(),
                        loss3.item()
                    )
            )
    return step


def adjust_lr(optimizer, epoch):
    if epoch == 200:
        lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save(model, optimizer, epoch, step, error, best_error):
    path = os.path.join(model_dir, '{:03}.ckpt'.format(epoch))
    # torch.save(model.state_dict(), path)
    # model.save_state_dict(path)

    state = {}
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['error'] = error
    state['epoch'] = epoch
    state['step'] = step

    torch.save(state, path)
    print('save model at epoch{}'.format(epoch))

    if error < best_error:
        best_error = error
        best_path = os.path.join(model_dir, 'best_model.ckpt'.format(epoch))
        shutil.copyfile(path, best_path)
        print('best model in epoch {}'.format(epoch))

    return best_error


if __name__ == '__main__':
    main(args)
    # writer.close()
