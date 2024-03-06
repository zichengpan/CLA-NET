# coding=utf8
from __future__ import division
import os

import cv2
import torch
import torch.utils.data as data
import pandas
import random
import PIL.Image as Image
from PIL import ImageStat
import numpy as np
from skimage.feature import local_binary_pattern, hog

def random_sample(img_names, labels):
    anno_dict = {}
    img_list = []
    anno_list = []
    for img, anno in zip(img_names, labels):
        if not anno in anno_dict:
            anno_dict[anno] = [img]
        else:
            anno_dict[anno].append(img)

    for anno in anno_dict.keys():
        anno_len = len(anno_dict[anno])
        fetch_keys = random.sample(list(range(anno_len)), anno_len//10)
        img_list.extend([anno_dict[anno][x] for x in fetch_keys])
        anno_list.extend([anno for x in fetch_keys])
    return img_list, anno_list

class dataset(data.Dataset):
    def __init__(self, Config, anno, swap_size=[2,2], common_aug=None, swap=None, totensor=None, train=False, train_val=False, test=False):
        self.Config = Config
        self.root_path = Config.rawdata_root
        self.numcls = Config.numcls
        self.dataset = Config.dataset
        self.use_cls_2 = Config.cls_2
        self.use_cls_mul = Config.cls_2xmul
        self.feature_enhance = Config.feature_enhance

        if isinstance(anno, pandas.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.labels = anno['label'].tolist()
        elif isinstance(anno, dict):
            self.paths = anno['img_name']
            self.labels = anno['label']

        if train_val:
            self.paths, self.labels = random_sample(self.paths, self.labels)
        self.common_aug = common_aug
        self.swap = swap
        self.totensor = totensor
        self.cfg = Config
        self.train = train
        self.swap_size = swap_size
        self.test = test

    def logm_ch(self, input):
        if self.feature_enhance:
            layer = 4
        else:
            layer = 3

        img = np.array(input, np.float32) / 255  # h w c
        img = img.transpose()

        img = img.reshape((layer, -1))
        img = np.transpose(img)
        img = torch.tensor(img)
        img = img - torch.mean(img, 0).expand_as(img)

        logm_img = self.cov_logm(img)
        # get up triangle
        index = torch.ones(layer, layer)
        # index = torch.triu(index)
        logm_img = logm_img[index.bool()]

        return logm_img

    def cov_logm(self, a):

        u, s, v = torch.svd(a)
        return v @ torch.diag(2 * s.log()) @ v.t()

    def lbp(self, x):

        imgUMat = np.float32(x)
        #
        gray = cv2.cvtColor(imgUMat, cv2.COLOR_RGB2GRAY)

        radius = 2
        n_points = 8 * radius

        METHOD = 'uniform'
        lbp = local_binary_pattern(gray, n_points, radius, METHOD)

        _, hog_img = hog(imgUMat, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

        # lbp = torch.from_numpy(lbp).long()

        return lbp, hog_img

    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])

        img = self.pil_loader(img_path)

        if self.test:
            img = self.totensor(img)
            label = self.labels[item] - 1
            return img, label, self.paths[item]

        if self.swap_size[1] == 2:
            foo = range(4)
        elif self.swap_size[1] == 3:
            foo = range(9)
        elif self.swap_size[1] == 4:
            foo = range(16)
        elif self.swap_size[1] == 6:
            foo = range(36)

        mask_num = random.sample(foo, self.Config.mask_num)  # masked number 1
        # print(mask_num)
        j = mask_num[0] % self.swap_size[0]
        i = mask_num[0] // self.swap_size[0]
        h_space = int(384/self.swap_size[0])
        w_space = int(384/self.swap_size[0])

        mask = np.ones([384, 384])
        mask[i * h_space: (i + 1) * h_space, j * w_space: (j + 1) * w_space] = 0
        mask = mask.reshape(384, 384, 1)

        img_unswap = self.common_aug(img) if not self.common_aug is None else img
        img_unswap_con = self.common_aug(img) if not self.common_aug is None else img

        tmp_ori = img_unswap
        tmp_con = img_unswap_con


        if self.feature_enhance:
            img_unswap = np.array(img_unswap)
            _, hog_layer = self.lbp(img_unswap)
            hog_layer = np.reshape(hog_layer, (384, 384, 1))
            img_unswap = np.concatenate((img_unswap, hog_layer), axis=2)
            img_unswap = Image.fromarray(np.uint8(img_unswap))

        img_unswap_mask = np.array(img_unswap) * mask
        img_unswap_con_mask = np.array(img_unswap_con) * mask

        img_unswap_mask = Image.fromarray(np.uint8(img_unswap_mask))
        img_unswap_con_mask = Image.fromarray(np.uint8(img_unswap_con_mask))

        image_unswap_list_original = self.crop_image(img_unswap, self.swap_size)  # from left to right, from top to bottom
        image_unswap_con_list_original = self.crop_image(img_unswap_con, self.swap_size)  # from left to right, from top to bottom

        covaMatrix_list_unswap_original = []  # original img cova
        covaMatrix_con_list_unswap_original = []  # original img cova

        for crop_img in image_unswap_list_original:

            covariance_matrix = self.cal_covariance(crop_img)
            covariance_matrix = self.logm_ch(covariance_matrix).numpy()
            covaMatrix_list_unswap_original.append(covariance_matrix)

        for crop_img in image_unswap_con_list_original:

            covariance_matrix = self.cal_covariance(crop_img)
            covariance_matrix = self.logm_ch(covariance_matrix).numpy()
            covaMatrix_con_list_unswap_original.append(covariance_matrix)

        image_unswap_list = self.crop_image(img_unswap_mask, self.swap_size)
        image_unswap_con_list = self.crop_image(img_unswap_con_mask, self.swap_size)

        covaMatrix_list_unswap = []  # original img with mask cova
        covaMatrix_con_list_unswap = []  # original img with mask cova

        for i_num, crop_img in enumerate(image_unswap_list):
            covariance_matrix = self.cal_covariance(crop_img)
            if i_num == mask_num[0]:  # nan
                if self.feature_enhance:
                    covaMatrix_list_unswap.append(np.zeros(16, dtype='float32'))
                    # covaMatrix_list_unswap.append(np.zeros((4, 4), dtype='float32'))
                else:
                    covaMatrix_list_unswap.append(np.zeros(9, dtype='float32'))
                    # covaMatrix_list_unswap.append(np.zeros((3, 3), dtype='float32'))
            else:
                covariance_matrix = self.logm_ch(covariance_matrix).numpy()
                covaMatrix_list_unswap.append(covariance_matrix)

        for i_num, crop_img in enumerate(image_unswap_con_list):
            covariance_matrix = self.cal_covariance(crop_img)
            if i_num == mask_num[0]:  # nan
                if self.feature_enhance:
                    covaMatrix_con_list_unswap.append(np.zeros(16, dtype='float32'))
                else:
                    covaMatrix_con_list_unswap.append(np.zeros(9, dtype='float32'))
            else:
                covariance_matrix = self.logm_ch(covariance_matrix).numpy()
                covaMatrix_con_list_unswap.append(covariance_matrix)

        img_unswap_mask = np.array(tmp_ori) * mask
        img_unswap_con_mask = np.array(tmp_con) * mask

        img_unswap_mask = Image.fromarray(np.uint8(img_unswap_mask))
        img_unswap_con_mask = Image.fromarray(np.uint8(img_unswap_con_mask))


        if self.train:
            img_con_swap = self.swap(img_unswap_con)  # Randomswap
            image_swap_con_list = self.crop_image(img_con_swap, self.swap_size)

            unswap_con_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_con_list_original]
            swap_con_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_con_list]

            covaMatrix_con_list_swap = []
            for swap_im in swap_con_stats:
                distance = [abs(swap_im - unswap_im) for unswap_im in unswap_con_stats]
                index = distance.index(min(distance))
                covaMatrix_con_list_swap.append(covaMatrix_con_list_unswap_original[index])

            # img_con_swap = np.array(img_con_swap) * mask
            # img_con_swap = Image.fromarray(np.uint8(img_con_swap))

            for mask_index in mask_num:
                if self.feature_enhance:
                    covaMatrix_con_list_swap[mask_index] = np.zeros(16, dtype='float32')
                else:
                    covaMatrix_con_list_swap[mask_index] = np.zeros(9, dtype='float32')

            covaMatrix_con_list_unswap_original = (np.array(covaMatrix_con_list_unswap_original).reshape(-1)).tolist()
            covaMatrix_con_list_swap = (np.array(covaMatrix_con_list_swap).reshape(-1)).tolist()
            covaMatrix_con_list_unswap = (np.array(covaMatrix_con_list_unswap).reshape(-1)).tolist()


            ###################### original ############################################

            img_swap = self.swap(img_unswap)  # Randomswap
            image_swap_list = self.crop_image(img_swap, self.swap_size)

            img_swap_con = self.swap(img_unswap)  # Randomswap

            unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list_original]
            swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]

            covaMatrix_list_swap = []
            for swap_im in swap_stats:
                distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
                index = distance.index(min(distance))
                covaMatrix_list_swap.append(covaMatrix_list_unswap_original[index])

            img_swap = np.array(img_swap) * mask
            img_swap = Image.fromarray(np.uint8(img_swap))

            img_swap_con = np.array(img_swap_con) * mask
            img_swap_con = Image.fromarray(np.uint8(img_swap_con))

            for mask_index in mask_num:

                if self.feature_enhance:
                    covaMatrix_list_swap[mask_index] = np.zeros(16, dtype='float32')
                    # covaMatrix_list_swap[mask_index] = np.zeros((4, 4), dtype='float32')

                else:
                    covaMatrix_list_swap[mask_index] = np.zeros(9, dtype='float32')
                    # covaMatrix_list_swap[mask_index] = np.zeros((3, 3), dtype='float32')

            covaMatrix_list_unswap_original = (np.array(covaMatrix_list_unswap_original).reshape(-1)).tolist()
            covaMatrix_list_swap = (np.array(covaMatrix_list_swap).reshape(-1)).tolist()
            covaMatrix_list_unswap = (np.array(covaMatrix_list_unswap).reshape(-1)).tolist()

            img_swap = self.totensor(img_swap)
            img_swap_con = self.totensor(img_swap_con)

            label = self.labels[item] - 1
            if self.use_cls_mul:
                label_swap = label + self.numcls
            if self.use_cls_2:
                label_swap = -1

            img_unswap = self.common_aug(img) if not self.common_aug is None else img
            img_unswap = self.totensor(img_unswap)
            img_unswap_mask = self.totensor(img_unswap_mask)

            img_unswap_con = self.common_aug(img) if not self.common_aug is None else img
            img_unswap_con = self.totensor(img_unswap_con)
            img_unswap_con_mask = self.totensor(img_unswap_con_mask)

            return img_unswap, img_unswap_mask, img_swap, label, label, label_swap, covaMatrix_list_unswap_original, \
                   covaMatrix_list_unswap, covaMatrix_list_swap, covaMatrix_con_list_unswap_original, covaMatrix_con_list_swap, \
                    covaMatrix_con_list_unswap, self.paths[item], img_unswap_con, img_unswap_con_mask, img_swap_con

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list


    def get_weighted_sampler(self):
        img_nums = len(self.labels)
        weights = [self.labels.count(x) for x in range(self.numcls)]
        return torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=img_nums)

    def cal_covariance(self, input):
        if self.feature_enhance:
            layer = 4
        else:
            layer = 3

        img = np.array(input,np.float32)/255  # h w c
        h,w,c = img.shape
        img = img.transpose((2, 0, 1))
        img = img.reshape((layer, -1))

        mean = img.mean(1)

        img = img -mean.reshape(layer, 1)

        covariance_matrix = np.matmul(img,np.transpose(img))
        covariance_matrix = covariance_matrix/(h*w-1)

        return covariance_matrix
        
# original, original mask, swap mask : 110
def collate_fn4train(batch):
    imgs = []
    imgs_con = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    imgs_con_second = []
    for sample in batch:

        imgs.append(sample[0])
        imgs.append(sample[1])
        imgs.append(sample[2])
        label.append(sample[3])
        label.append(sample[3])
        label.append(sample[3])
        if sample[5] == -1:  # 110
            label_swap.append(1)
            label_swap.append(1)
            label_swap.append(0)
        else:
            label_swap.append(sample[2])
            label_swap.append(sample[3])
        law_swap.append(sample[6])
        law_swap.append(sample[7])
        law_swap.append(sample[8])
        img_name.append(sample[12])

        imgs_con.append(sample[9])
        imgs_con.append(sample[10])
        imgs_con.append(sample[11])

        imgs_con_second.append((sample[13]))
        imgs_con_second.append((sample[14]))
        imgs_con_second.append((sample[15]))

    return torch.stack(imgs, 0), label, label_swap, law_swap, imgs_con, torch.stack(imgs_con_second, 0), img_name

def collate_fn4val(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        if sample[3] == -1:
            label_swap.append(1)
        else:
            label_swap.append(sample[2])
        law_swap.append(sample[3])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name

def collate_fn4backbone(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        if len(sample) == 5:
            label.append(sample[1])
        else:
            label.append(sample[2])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name


def collate_fn4test(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name
