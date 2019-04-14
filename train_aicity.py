from __future__ import absolute_import
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
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
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=20, type=int,
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
#parser.add_argument('-rf','--return-features',default=False)
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=5,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=2, help="start to evaluate after specific epoch")
parser.add_argument('--use-metric-cuhk03',action='store_true',help="whether to use cuhk-03 metric(default:False)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--reranking',action= 'store_true', help= 'result re_ranking')

parser.add_argument('--test_distance',type = str, default='global', help= 'test distance type')
parser.add_argument('--unaligned',action= 'store_true', help= 'test local feature with unalignment')
# pcb settings
parser.add_argument('--share_conv', action='store_true',default=False)
parser.add_argument('--stripes', type=int, default=4)
parser.add_argument('--train_all',action='store_true',default=False)
parser.add_argument('--use_pcb',action='store_true',default=False)

args = parser.parse_args()

def main():
    use_gpu = torch.cuda.is_available()
#    use_gpu = False
    if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False

    if not args.evaluate:
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
        T.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
#        T.RandomVerticalFlip(),
#        T.RandomRotation(30),
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
    #embed() 
    print('len of trainloader',len(trainloader))
    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print('len of queryloader',len(queryloader))
    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print('len of galleryloader',len(galleryloader))
    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_vids, num_stripes=args.stripes, share_conv=args.share_conv,use_pcb=args.use_pcb)
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

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return 0

    start_time = time.time()
    train_time = 0
    best_mAP = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        if args.stepsize > 0: scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch or ((epoch+1)==1):
            print("==> Test")
            mAP = test(model, queryloader, galleryloader, use_gpu)
            is_best = mAP > best_mAP

            if is_best:
                best_mAP = mAP
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'mAP': mAP,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
    
    print("==> Best mAP {:.2%}, achieved at epoch {}".format(best_mAP, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    xent_losses = AverageMeter()
    global_losses = AverageMeter()
    local_losses = AverageMeter()

    end = time.time()
    for batch_idx, (imgs, pids) in enumerate(trainloader):    
#        print('pids',pids.shape)
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)
#        outputs, features, local_features = model(imgs)
        #embed()
        if args.use_pcb:
            outputs,gf_out,local_features,features = model(imgs)
            #embed()
        elif not args.use_pcb:
            # only use global feature to get the classifier results
            outputs= model(imgs)
        # print('outputs',(outputs.shape))
        # htri_only = False(default)
        if args.htri_only:
            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
            else:
                # print ('pids:', pids)
                global_loss, local_loss = criterion_metric(features, pids, local_features)
        else:
            if isinstance(outputs, tuple):
                xent_loss = DeepSupervision(criterion_class, outputs, pids)
            else:
                if args.use_pcb:
                    xent_loss = 0.0
                    for logits in outputs:
                        stripe_loss = criterion_class(logits, pids)
                        xent_loss += stripe_loss
                elif not args.use_pcb:
                    xent_loss = criterion_class(outputs, pids)
                if isinstance(features, tuple):
                    global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
                else:
                    global_loss, local_loss = criterion_metric(features, pids, local_features)
        
        loss = xent_loss + global_loss + local_loss
        # loss = global_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), pids.size(0))
        xent_losses.update(xent_loss.item(), pids.size(0))
        global_losses.update(global_loss.item(), pids.size(0))
        local_losses.update(local_loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLoss {xent_loss.val:.4f} ({xent_loss.avg:.4f})\t'
                  'GLoss {global_loss.val:.4f} ({global_loss.avg:.4f})\t'
                  'LLoss {local_loss.val:.4f} ({local_loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,data_time=data_time,
                   loss=losses,xent_loss=xent_losses, global_loss=global_losses, local_loss = local_losses))


def extract_feature(model, inputs, requires_norm, vectorize, requires_grad=False,use_pcb=False):

    # Move to model's device
    #print('inputs',inputs.shape)
    inputs = inputs.to(next(model.parameters()).device)

    with torch.set_grad_enabled(requires_grad):
        if use_pcb:
            features,global_f = model(inputs)
            size = features.shape
            if requires_norm:
        # [N, C*H]
                features = features.view(size[0], -1)

        # norm feature
                fnorm = features.norm(p=2, dim=1)
                features = features.div(fnorm.unsqueeze(dim=1))

            if vectorize:
                features = features.view(size[0], -1)
            else:
            # Back to [N, C, H=S]
                features = features.view(size)
            features = torch.cat((features,global_f),-1)
            return features
        elif not use_pcb:
            features = model(inputs)
            size = features.shape
            if requires_norm:
        # [N, C*H]
                features = features.view(size[0], -1)

        # norm feature
                fnorm = features.norm(p=2, dim=1)
                features = features.div(fnorm.unsqueeze(dim=1))

            if vectorize:
                features = features.view(size[0], -1)
            else:
            # Back to [N, C, H=S]
                features = features.view(size)
            return features


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        for batch_idx, (imgs, pids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            qf.append(extract_feature(
                model, imgs, requires_norm=True, vectorize=True,use_pcb=args.use_pcb).cpu().data)
            #_,local_features,features = model(imgs)
            #print('lqf shape',local_features.shape)
            #print('qf shape',features.shape)
            batch_time.update(time.time() - end)
            
            #features = features.data.cpu()
            #local_features = local_features.data.cpu()
            #qf.append(features)
            #lqf.append(local_features)
            q_pids.extend(pids)
            #q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        #lqf = torch.cat(lqf,0)
        print('qf shape',qf.shape)
        #print('lqf shape',lqf.shape)
        q_pids = np.asarray(q_pids)
        #q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, lgf = [], [], [], []
        end = time.time()
        for batch_idx, (imgs, pids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            # features, local_features = model(imgs)
            gf.append(extract_feature(
               model, imgs, requires_norm=True, vectorize=True,use_pcb=args.use_pcb).cpu().data)
            #_,local_features,features = model(imgs)
            batch_time.update(time.time() - end)

#            features = features.data.cpu()
#            local_features = local_features.data.cpu()
#            gf.append(features)
#            lgf.append(local_features)
            g_pids.extend(pids)
#            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
#        lgf = torch.cat(lgf,0)
        print('gf shape',gf.shape)
#        print('lgf shape',lgf.shape)
        g_pids = np.asarray(g_pids)
#        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))
    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    print('distmat shape1',distmat.shape)
    distmat.addmm_(1, -2, qf, gf.t())
    print('distmat shape2',distmat.shape)
    distmat = distmat.cpu().numpy()
    
    # args.test_distance = 'global'(default) 
    if not args.test_distance== 'global':
        print("Only using global branch")
        from util.distance import low_memory_local_dist
        #embed()
        lqf = lqf.permute(0,2,1)
        lgf = lgf.permute(0,2,1)
        local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(),aligned= not args.unaligned)
        if args.test_distance== 'local':
            print("Only using local branch")
            distmat = local_distmat
        if args.test_distance == 'global_local':
            print("Using global and local branches")
            distmat = local_distmat+distmat
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    # args.reranking = false(default) 
    if args.reranking:
        from util.re_ranking import re_ranking
        if args.test_distance == 'global':
            print("Only using global branch for reranking")
            distmat = re_ranking(qf,gf,k1=20, k2=6, lambda_value=0.3)
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
        print("Computing CMC and mAP for re_ranking")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

        print("Results ----------")
        print("mAP(RK): {:.1%}".format(mAP))
        print("CMC curve(RK)")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
    return mAP

if __name__ == '__main__':
    main()
