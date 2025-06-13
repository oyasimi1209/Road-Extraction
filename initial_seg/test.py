import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
import argparse
import json
from time import time
from datetime import datetime

from dlinknet import DLinkNet34, MyFrame, dice_bce_loss
from datainput import ImageFolder

BATCHSIZE_PER_CARD = 2


def train(strDataPath, strModelPath, strOutPath):
    print("......Training......")
    tic = time()
    timestamp = datetime.fromtimestamp(tic).strftime('%Y%m%d-%H%M')
    NAME = f'road_{timestamp}.th'
    print(f"Model will be saved as: {NAME}")
    SHAPE = (1024, 1024)
    
    # Convert relative paths to absolute paths if needed
    if not os.path.isabs(strDataPath):
        strDataPath = os.path.abspath(strDataPath)
    if not os.path.isabs(strModelPath):
        strModelPath = os.path.abspath(strModelPath)
    if not os.path.isabs(strOutPath):
        strOutPath = os.path.abspath(strOutPath)
    
    sat_dir = os.path.join(strDataPath, 'train_satellite')
    lab_dir = os.path.join(strDataPath, 'train_label')
    
    # Check if directories exist
    if not os.path.exists(sat_dir):
        raise FileNotFoundError(f"Satellite images directory not found: {sat_dir}")
    if not os.path.exists(lab_dir):
        raise FileNotFoundError(f"Label images directory not found: {lab_dir}")
        
    print(f"Using satellite images from: {sat_dir}")
    print(f"Using label images from: {lab_dir}")
    
    # Get all satellite image files and extract their base names
    imagelist = [f for f in os.listdir(sat_dir) if f.endswith('_sat.png')]
    trainlist = [f[:-8] for f in imagelist]  # Remove '_sat.png' to get base name

    solver = MyFrame(DLinkNet34, dice_bce_loss, 5e-4)
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

    dataset = ImageFolder(trainlist, sat_dir, lab_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)
    if os.path.isdir(strOutPath + '/logs/'):
        pass
    else:
        os.makedirs(strOutPath + '/logs/')

    mylog = open(strOutPath + '/logs/' + NAME + '.log', 'w')
    no_optim = 0
    total_epoch = 300
    train_epoch_best_loss = 100.
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)
        print('********', file=mylog)
        print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
        print('train_loss:', train_epoch_loss, file=mylog)
        print('SHAPE:', SHAPE, file=mylog)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', SHAPE)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            model_path = os.path.join(strModelPath, NAME)
            solver.save(model_path)
            print(f"Saved model to: {model_path}")
        if no_optim > 6:
            print('early stop at %d epoch' % epoch, file=mylog)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load(strModelPath + '/' + NAME + '.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)
        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
            
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


def test_images():
    # 使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(base_dir)
    
    # 配置路径
    test_data_dir = os.path.join(workspace_dir, "data", "Shaoxing", "test_satellite")
    model_path = os.path.join(workspace_dir, "model", "road_20250610-0009.th")  # 替换为你的模型文件名
    output_dir = os.path.join(workspace_dir, "output", "test_results")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing model...")
    solver = TTAFrame(DLinkNet34)
    
    print(f"Loading model from: {model_path}")
    solver.load(model_path)
    
    print(f"Processing test images from: {test_data_dir}")
    test_images = [f for f in os.listdir(test_data_dir) if f.endswith('_sat.png')]
    
    tic = time()
    for i, image_name in enumerate(test_images):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(test_images)}, Time: {time()-tic:.2f}s")
            
        # 处理图片
        image_path = os.path.join(test_data_dir, image_name)
        mask = solver.test_one_img_from_path(image_path)
        
        # 二值化处理
        mask_binary = mask.copy()
        mask_binary[mask_binary > 4] = 255
        mask_binary[mask_binary <= 4] = 0
        
        # 保存结果
        output_name = image_name.replace('_sat.png', '_mask.png')
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, mask_binary.astype(np.uint8))
    
    print(f"\nProcessing complete! Results saved to: {output_dir}")
    print(f"Total time: {time()-tic:.2f}s")


if __name__ == "__main__":
    test_images()