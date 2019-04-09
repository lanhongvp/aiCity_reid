import random
import os
import os.path as osp

from PIL import Image
from torchvision import  transforms
from torch.autograd import Variable
import torch

from IPython import embed
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import preprocessing as pre
from util.re_ranking import re_ranking

def get_track_id(root_dir,is_train=False):
    dataset_dir = osp.abspath(osp.join(root_dir,'aiCity'))
    gallery_track_name = 'test_track_id.txt'
    train_track_name = 'train_track_id.txt'
    gallery_track_dir = osp.join(dataset_dir,gallery_track_name)
    train_track_dir = osp.join(dataset_dir,train_track_name)
    print('gallery_dir',gallery_track_dir)
    print('train_dir',train_track_dir)
    
    if is_train:
        pass
    elif not is_train:
        gt_ids = {}
        with open(gallery_track_dir,'r') as gallery_track_f:
            gt_idx = 0  
            for gallery_track_id in gallery_track_f.readlines():
                gallery_tid = gallery_track_id.split()
                gt_ids[gt_idx] = list(map(lambda gt_id:int(gt_id),gallery_tid))
                gt_idx += 1
        return gt_ids

def track_info_average(track_id,global_f,local_f):
    # track_id: dcit
    # global_f::torch.tensor
    #local_f: torch.tensor 
    global_ft = torch.zeros((len(track_id),global_f.shape[1]))
    local_ft = torch.zeros_like(global_ft)
    t_id_st = 0
    for t_idx,t_id in enumerate(track_id):
        t_id_len = len(list(track_id[t_idx]))
        t_id_end = t_id_st + t_id_len
        global_ft[t_idx] = torch.sum(global_f[t_id_st:t_id_end,:],dim=0,keepdim=True)/t_id_len
        #  local_ft[t_idx] = torch.sum(local_f[t_id_st:t_id_end,:],dim=0,keepdim=True)/t_id_len
        t_id_st = t_id_end
    return global_ft,local_ft

    
def test_rank100_aicity(distmat,q_imgs,g_imgs,track_id,use_track_info=False,rank_k=100):
    q_num = distmat.shape[0]
    g_num = distmat.shape[1]

    test_rank_result = {}
    # key: q_id
    # value: k from 1 to 100
    idx = 0

    for q_id in q_imgs:
        test_rank_tmp = []
        distmat_cmp = np.argsort(distmat[idx,:])
        distmat_rank_k = distmat_cmp[:rank_k]
        if not use_track_info:
#        embed()
            for k in range(rank_k):
                test_rank_tmp.append((g_imgs[distmat_rank_k[k]],k+1))
            test_rank_result[q_id] = test_rank_tmp
            idx = idx+1
#        embed()
        elif use_track_info:
            cnt = 0 
            for k in range(rank_k):
                test_rank_tmp += list(map(lambda t_id:(t_id,k+1),list(track_id[distmat_rank_k[k]])))
                cnt += len(list(track_id[distmat_rank_k[k]]))
                if cnt >= rank_k:
                    test_rank_result[q_id] = test_rank_tmp
                    idx = idx + 1
                    break
            #embed()
    return test_rank_result

def calacc(probeFeat, probeLabel, galleryFeat, galleryLabel, rerank=False, top_num=[1,5,10,20,30]):
    if rerank:
        dist = re_ranking(probeFeat,galleryFeat,k1=10,k2=6,lambda_value=0.3)
    else:
        dist = cdist(probeFeat, galleryFeat)
    index = []
    for i in range(len(dist)):
        a = dist[i]
        ind = np.where(galleryLabel == probeLabel[i])
        dp = a[ind]
        a.sort()
        index.append(list(a).index(dp))

    index = np.array(index)
    cmc = lambda top,index : float(len(np.where(index<top)[0]))/len(probeLabel)
    cmc_curve = [cmc(top, index) for top in top_num]
    return cmc_curve

def cmc_curve(probeFeat, probeLabel, galleryFeat, galleryLabel, test_num=1, norm_flag = True, rerank = False):
    cmc_result = []

    for i in range(test_num):
        cmc_result.append(calacc(probeFeat, probeLabel, galleryFeat, galleryLabel,norm_flag,rerank))
        print('test time: ', i)
    cmc_mean = np.mean(np.array(cmc_result),axis=0)
    return  cmc_mean

def calmap(probeFeat, probeLabel, galleryFeat, galleryLabel, norm_flag=True, rerank=False):
    if norm_flag:
        probeFeat = pre.normalize(probeFeat,axis=1)
        galleryFeat = pre.normalize(galleryFeat,axis=1)
    if rerank:
        dist = re_ranking(probeFeat,galleryFeat,k1=10,k2=6,lambda_value=0.3)
    else:
        dist = cdist(probeFeat, galleryFeat)
    ap = []
    for i in range(len(dist)):
        a = dist[i]
        rank = []
        ind = list(np.where(galleryLabel == probeLabel[i])[0])
        dp = a[ind]
        a.sort()
        for j in dp:
            rank.append(list(a).index(j)+1)
        rank.sort()
        thisap = 0.
        for k in range(len(ind)):
            thisap = thisap + float((k+1))/rank[k]
        ap.append(thisap/len(ind))
    ap = np.array(ap)
    map = np.mean(ap)

    return map
