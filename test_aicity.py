from __future__ import absolute_import
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import models
from util.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID
from util import data_manager
from util import transforms as T
from util.dataset_loader import ImageDataset,ImageDatasetCa
from util.utils import Logger
from util.utils import AverageMeter, Logger, save_checkpoint
from util.eval_metrics import evaluate
from util.optimizers import init_optim
from util.samplers import RandomIdentitySampler
from util.test_CMC import test_rank100_aicity,track_info_average,get_track_id
from IPython import embed
#zhushi

parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')
# Datasets
parser.add_argument('--root', type=str, default='../data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='aiCityVeRi',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")

# Optimization options
parser.add_argument('--labelsmooth', action='store_true', help="label smooth")
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=300, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=150, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# triplet hard loss
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('-rf','--return-features',default=False)
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--test', action='store_true', help="test aicity dataset")
parser.add_argument('--result_dir',type=str,default='exp')
parser.add_argument('--eval-step', type=int, default=5,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=2, help="start to evaluate after specific epoch")
parser.add_argument('--use-metric-cuhk03',action='store_true',help="whether to use cuhk-03 metric(default:False)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--reranking',action= 'store_true', help= 'result re_ranking')
parser.add_argument('--use_track_info',action='store_true',help='whether to use track info')

parser.add_argument('--test_distance',type = str, default='global', help= 'test distance type')
parser.add_argument('--unaligned',action= 'store_true', help= 'test local feature with unalignment')

parser.add_argument('--share_conv', action='store_true',default=False)
parser.add_argument('--stripes', type=int, default=4)

args = parser.parse_args()

def main():
    use_gpu = torch.cuda.is_available()
#    use_gpu = False
    if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False

    if not args.test:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
    )

    print('dataset',dataset)
    # data augmentation
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        # T.Resize(size=(384,128),interpolation=3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        #T.Resize(size=(384,128),interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
		batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )
    
    print('len of trainloader',len(trainloader))
    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test,train=False),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print('len of queryloader',len(queryloader))
    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test,train=False),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )
    #embed()
    print('len of galleryloader',len(galleryloader))
    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_vids,
                            loss={'softmax','metric'}, aligned =True, use_gpu=use_gpu)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    print('Model ',model)
    print('num_classes',dataset.num_train_vids)
    if args.labelsmooth:
        criterion_class = CrossEntropyLabelSmooth(num_classes=dataset.num_train_vids, use_gpu=use_gpu)
    else:
        # criterion_class = CrossEntropyLoss(use_gpu=use_gpu)
        criterion_class = nn.CrossEntropyLoss()
    criterion_metric = TripletLossAlignedReID(margin=args.margin)
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.test:
        print("test aicity dataset")
        if args.use_track_info:
            g_track_id = get_track_id(args.root)
            test(model, queryloader, galleryloader, use_gpu,dataset_q=dataset.query,dataset_g=dataset.gallery,track_id_tmp=g_track_id,rank=100)
        else:
            test(model, queryloader, galleryloader, use_gpu,dataset_q=dataset.query,dataset_g=dataset.gallery,rank=100)
        return 0

	
def test(model, queryloader, galleryloader, use_gpu, dataset_q,dataset_g,track_id_tmp=None,rank=100):
    batch_time = AverageMeter()
    #embed()
    model.eval()
    
    with torch.no_grad():
        qf, lqf, q_imgs = [], [], []

        for q_idx in range(len(dataset_q)):
            q_img = int(dataset_q[q_idx].split('/')[-1].strip('.jpg'))
            q_imgs.append(q_img)
    
        for batch_idx, (imgs) in enumerate(queryloader):
            #embed()
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features,local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf,0)
        print('lqf shape',lqf.shape)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, lgf, g_imgs = [], [], []

        for g_idx in range(len(dataset_g)):
            g_img = int(dataset_g[g_idx].split('/')[-1].strip('.jpg'))
            g_imgs.append(g_img)
        end = time.time()
        
        # embed()
        #obtain the track infoi
        for batch_idx, (imgs) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features,local_features = model(imgs)
            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)

        #embed()
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf,0)
        print('lgf shape',lgf.shape)
        gt_f,_ = track_info_average(track_id_tmp,gf,lgf)
        embed()
        print('len of gimgs',len(g_imgs))
        print('Extracted features for gallery_track set,obtained {}-by-{} matrix'.format(gt_f.size(0),gt_f.size(1)))
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    #embed()
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))
    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    #gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    gt_f = 1. * gt_f / (torch.norm(gt_f, 2, dim = -1, keepdim=True).expand_as(gt_f) + 1e-12)
    
    m, n = qf.size(0), gt_f.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gt_f, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gt_f.t())
    distmat = distmat.numpy()
    
    #embed()
    print("------------------")

    if args.reranking:
        from util.re_ranking import re_ranking
        if args.test_distance == 'global':
            print("Only using global branch for reranking")
            distmat = re_ranking(qf,gt_f,k1=20, k2=6, lambda_value=0.3)
        else:
            local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(),aligned= not args.unaligned)
            local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(),aligned= not args.unaligned)
            local_dist = np.concatenate(
                [np.concatenate([local_qq_distmat, local_distmat], axis=1),
                 np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
                axis=0)
            if args.test_distance == 'local':
                print("Only using local branch for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=True)
            elif args.test_distance == 'global_local':
                print("Using global and local branches for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=False)
    #embed()
    print("Computing CMC and mAP for re_ranking")

    print("==> Test aicity dataset and write to csv")
    test_rank_result = test_rank100_aicity(distmat,q_imgs,g_imgs,track_id_tmp,use_track_info=True)
    # test_rank_result is a dict, use pandas to convert 
    # embed()
    test_rank_result_df = pd.DataFrame(list(test_rank_result.items()),columns=['query_ids','gallery_ids'])
    test_result_df = test_rank_result_df.sort_values('query_ids')
    # write to csvi
    embed()
    with open('aic_res_'+args.result_dir+'.txt','w') as f:
        for idx in range(len(test_result_df)):
            sep_c = ' '
            row_ranks = []
            idx_row = test_result_df.iloc[idx]['gallery_ids'][:100]
            #embed()
            for item in idx_row:
                row_rank = str(item[0])
                row_ranks.append(row_rank)
            sep_c = sep_c.join(row_ranks)
            #embed()
            sep_c = sep_c+'\n'
            #embed()
            f.write(sep_c)
        f.close()

   
if __name__ == '__main__':
    main()
