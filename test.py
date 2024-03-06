#coding=utf-8
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import numpy as np
from math import ceil
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models.LoadModel import MainModel
from utils.dataset import collate_fn4train, collate_fn4test, collate_fn4val, dataset
from config import LoadConfig, load_data_transformers
from utils.test_tool import set_text, save_multi_img, cls_base_acc
from sklearn.manifold import TSNE
import random

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['cuda_visible_devices'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='Test CDRM parameters')
    parser.add_argument('--data', dest='dataset',
                        default='STCAR', type=str,
                        help="COTTON, Soybean200, Soybean2000, soybean_gene, R1, R3, R4, R5, R6")
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=16, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=16, type=int)
    parser.add_argument('--ver', dest='version',
                        default='val', type=str)
    parser.add_argument('--save', dest='resume',
                        default='./logs/STCAR/cub_model1_2*2/savepoint-last-model-steps203600-valtop1acc0.9248_valtop3acc0.9826.pth', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=440, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=384, type=int)
    parser.add_argument('--ss', dest='save_suffix',
                        default=None, type=str)
    parser.add_argument('--acc_report', dest='acc_report',
                        action='store_true')
    parser.add_argument('--swap_num', default=[2, 2],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args



def scale_to_01_range(x):

    value_range = (np.max(x) - np.min(x))

    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

if __name__ == '__main__':
    args = parse_args()
    print(args)

    Config = LoadConfig(args, args.version)
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)

    data_set = dataset(Config,
                       anno=Config.val_anno if args.version == 'val' else Config.test_anno,
                       swap_size=args.swap_num,
                       common_aug=transformers["None"],
                       swap=transformers["None"],  # transformers["swap"], #
                       totensor=transformers['test_totensor'],
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=collate_fn4test)

    setattr(dataloader, 'total_item_len', len(data_set))

    cudnn.benchmark = True

    model = MainModel(Config, args)
    model_dict=model.state_dict()
    resume = args.resume
    pretrained_dict=torch.load(resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model = nn.DataParallel(model)

    model.train(False)

    with torch.no_grad():
        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0
        val_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        count_bar = tqdm(total=dataloader.__len__())

        embedding_list = []
        label_list = []

        for batch_cnt_val, data_val in enumerate(dataloader):
            count_bar.update(1)
            inputs, labels, img_name = data_val
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

            outputs = model(inputs)

            outputs_pred = outputs[0]

            top3_val, top3_pos = torch.topk(outputs_pred, 3)

            embedding_list.append(outputs[3].cpu())
            label_list.append(labels.cpu())

            if args.version == 'val':
                batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
                val_corrects1 += batch_corrects1
                batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
                val_corrects2 += (batch_corrects2 + batch_corrects1)
                batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
                val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

            if args.acc_report:
                for sub_name, sub_cat, sub_val, sub_label in zip(img_name, top3_pos.tolist(), top3_val.tolist(), labels.tolist()):
                    result_gather[sub_name] = {'top1_cat': sub_cat[0], 'top2_cat': sub_cat[1], 'top3_cat': sub_cat[2],
                                               'top1_val': sub_val[0], 'top2_val': sub_val[1], 'top3_val': sub_val[2],
                                               'label': sub_label}

    if args.acc_report:
        torch.save(result_gather, 'result_gather_%s'%resume.split('/')[-1][:-4]+ '.pt')

    count_bar.close()

    val_acc1 = val_corrects1 / len(data_set)
    val_acc2 = val_corrects2 / len(data_set)
    val_acc3 = val_corrects3 / len(data_set)
    print('%sacc1 %f%s n%sacc2 %f%s\n%sacc3 %f%s\n'%(8*'-', val_acc1, 8*'-', 8*'-', val_acc2, 8*'-', 8*'-',  val_acc3, 8*'-'))

    # if args.acc_report:

        # val_acc1 = val_corrects1 / len(data_set)
        # val_acc2 = val_corrects2 / len(data_set)
        # val_acc3 = val_corrects3 / len(data_set)
        # print('%sacc1 %f%s n%sacc2 %f%s\n%sacc3 %f%s\n'%(8*'-', val_acc1, 8*'-', 8*'-', val_acc2, 8*'-', 8*'-',  val_acc3, 8*'-'))

        # cls_top1, cls_top3, cls_count = cls_base_acc(result_gather)

        # acc_report_io = open('acc_report_%s_%s.json'%(args.save_suffix, resume.split('/')[-1]), 'w')
        # json.dump({'val_acc1':val_acc1,
        #            'val_acc2':val_acc2,
        #            'val_acc3':val_acc3,
        #            'cls_top1':cls_top1,
        #            'cls_top3':cls_top3,
        #            'cls_count':cls_count}, acc_report_io)
        # acc_report_io.close()


