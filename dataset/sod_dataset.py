import os
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data.distributed import DistributedSampler

class NormalDataset(Dataset):
    def __init__(self, data_path, dataset, transform, mode = 'train', local_rank = 0, max_rank = 1):
        self.imgs_list = []
        self.masks_list = []

        if dataset == 'rdvs':
            lable_rgb = 'rgb'
            lable_depth = 'Depth'
            lable_gt = 'ground-truth'
            lable_flow = 'FLOW'

            if mode == 'train':
                data_dir = os.path.join(data_path, 'RDVS/train')
            else:
                data_dir = os.path.join(data_path, 'RDVS/test')
        elif dataset == 'vidsod_100':
            lable_rgb = 'rgb'
            lable_depth = 'depth'
            lable_gt = 'gt'
            lable_flow = 'flow'
            
            if mode == 'train':
                data_dir = os.path.join(data_path, 'vidsod_100/train')
            else:
                data_dir = os.path.join(data_path, 'vidsod_100/test')
        elif dataset == 'dvisal':
            lable_rgb = 'RGB'
            lable_depth = 'Depth'
            lable_gt = 'GT'
            lable_flow = 'flow'

            data_dir = os.path.join(data_path, 'DViSal_dataset/data')

            if mode == 'train':
                dvi_mode = 'train'
            else:
                dvi_mode = 'test_all'
        else:
            raise 'dataset is not support now.'
        
        if dataset == 'dvisal':
            with open(os.path.join(data_dir, '../', dvi_mode+'.txt'), mode='r') as f:
                subsets = set(f.read().splitlines())
        else:
            subsets = os.listdir(data_dir)
        
        for video in subsets:
            video_path = os.path.join(data_dir, video)
            rgb_path = os.path.join(video_path, lable_rgb)
            depth_path = os.path.join(video_path, lable_depth)
            gt_path = os.path.join(video_path, lable_gt)
            flow_path = os.path.join(video_path, lable_flow)
            frames = os.listdir(rgb_path)
            frames = sorted(frames)
            for frame in frames[:-1]:
                img_file_path = os.path.join(rgb_path, frame)
                if os.path.isfile(img_file_path):
                    self.imgs_list.append(img_file_path)
                    self.masks_list.append(os.path.join(gt_path, frame.replace('jpg', 'png')))

        self.transform=transform

        self.mode = mode

        if mode != 'train':
            # In order to enable distributed inference across multiple GPUs
            start_idx = (int)(local_rank * len(self.imgs_list) / max_rank)
            end_idx = (int)(local_rank * len(self.imgs_list) / max_rank + len(self.imgs_list) / max_rank)
            self.imgs_list = self.imgs_list[start_idx: end_idx]
            self.masks_list = self.masks_list[start_idx: end_idx]
    
    def __len__(self):
        return len(self.imgs_list)
    
    
    def __getitem__(self,index):
        img_dir= self.imgs_list[index]
        mask_dir= self.masks_list[index]

        #print(img_dir, mask_dir)

        mask_name = mask_dir.split("/")[-1]

        img=cv2.imread(img_dir)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        mask=cv2.imread(mask_dir)
        mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

        # load the original resolution mask for calculating metrics when inferencing
        if not self.mode == 'train':
            ori_mask = torch.from_numpy(mask) / 255.0
            
        augmented=self.transform(image=img,mask=mask)
        img = augmented['image']
        mask = augmented['mask'] / 255.0

        if self.mode == 'train':
            return {'img':img, 'mask':mask}
        else:
            return {'img':img, 'mask':mask, 'ori_mask':ori_mask, 'mask_name':mask_name}

def get_augmentation(version=0, img_size = 512):
    if version==0:
        transforms=albu.Compose([
            albu.OneOf([
                    albu.HorizontalFlip(),
                    albu.VerticalFlip(),
                    albu.RandomRotate90()
                ], p=0.5),
            albu.OneOf([
                    albu.MotionBlur(blur_limit=5),
                    albu.MedianBlur(blur_limit=5),
                    albu.GaussianBlur(blur_limit=5),
                    albu.GaussNoise(var_limit=(5.0, 20.0)),
                ], p=0.5),
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transforms=albu.Compose([albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    return transforms


def getSODDataloader(data_path, dataset, batch_size, num_workers, mode, local_rank = 0, max_rank = 1, img_size = 512):
    if mode == "train":
        transform = get_augmentation(0, img_size)
        dataset = NormalDataset(data_path, dataset, transform, mode)
        sampler = DistributedSampler(dataset)
        dataLoader = DataLoader(dataset,batch_size = batch_size, sampler = sampler, num_workers = num_workers)
    else:
        transform = get_augmentation(1, img_size)
        # max_rank represents the number of GPUs used for inference
        # local_rank represents the GPU currently in use.
        dataset = NormalDataset(data_path, dataset, transform, mode, local_rank, max_rank)
        dataLoader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers)
    return dataLoader