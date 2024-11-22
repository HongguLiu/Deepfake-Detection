# """
# Author: Honggu Liu
#
# """
#
# from PIL import Image
# from torch.utils.data import Dataset
# import os
# import random
#
#
# class MyDataset(Dataset):
#     def __init__(self, txt_path, transform=None, target_transform=None):
#         fh = open(txt_path, 'r')
#         imgs = []
#         for line in fh:
#             line = line.rstrip()
#             words = line.split()
#             imgs.append((words[0], int(words[1])))
#
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         fn, label = self.imgs[index]
#         img = Image.open(fn).convert('RGB')
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, label
#
#     def __len__(self):
#         return len(self.imgs)

"""
Author: Honggu Liu
"""
from PIL import Image
from torch.utils.data import Dataset, random_split
import os
import random


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None, split_ratio=0.7, is_train=True):
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = []

        # 텍스트 파일에서 데이터를 읽어와 이미지와 레이블을 리스트에 저장
        with open(txt_path, 'r') as fh:
            for line in fh:
                line = line.rstrip()
                words = line.split()
                self.imgs.append((words[0], int(words[1])))

        # 데이터셋을 70% 학습용, 30% 검증용으로 나눕니다.
        split_index = int(len(self.imgs) * split_ratio)

        # 학습과 검증 데이터에 따라 나누어진 데이터를 할당
        if is_train:
            self.imgs = self.imgs[:split_index]  # 70% 학습 데이터
        else:
            self.imgs = self.imgs[split_index:]  # 30% 검증 데이터

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
