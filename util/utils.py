from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import array,argmin
import pickle
import torch
from shutil import copyfile
from IPython import embed

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def dict_key_slice(ori_dict, start, end):
    """
    dict slice according to the key
    :param ori_dict: original dict
    :param start: start idx
    :param end: end idx
    :return: slice dict
    """
    if end != -1:
        slice_dict = {k: ori_dict[k] for k in list(ori_dict.keys())[start:end]}
    else:
        slice_dict = {k: ori_dict[k] for k in list(ori_dict.keys())[start:]}
    return slice_dict


def dict_value_slice(ori_dict,st,ed):
    """
    dict slice according to the value
    the original dict value could be sliced
    :param ori_dict: original dict
    :param st: start idx
    :param ed: end idx
    :return: slice dict
    """
    slice_dict = {}
    for item in ori_dict.items():
        vid = item[0]
        vnames = item[1]
        tmp_name = []
        if ed==-1:
            for i in range(st,len(vnames)):
                tmp_name.append(vnames[i])
        else:
            for i in range(st,ed):
                tmp_name.append(vnames[i])
        slice_dict[vid] = tmp_name
    return slice_dict


def write_pickle_aicity(download_path):
    if not os.path.isdir(download_path):
        print('please change the download_path')

    train_label = download_path + '/train_label.csv'
    tnames = {}
    tnames_p = open('aicity_train.pkl','wb')

    # for root, dirs, files in os.walk(train_path, topdown=True):
    with open(train_label,'r') as f:
        for line in f.readlines():
            tname = line.strip('\n').split(',')
            vid = tname[0]
            timg = [tname[1]] 
            if vid in tnames:
                tnames[vid] += timg
            else:
                tnames[vid] = timg
        pickle.dump(tnames,tnames_p)
        tnames_p.close()
        f.close()

def write_pickle_veri(download_path,pickle_name):
    # if not os.path.isdir(download_path):
    #     print('please change the download_path')

    train_label = download_path
    tnames = {}
    tnames_p = open(pickle_name+'.pkl','wb')

    # for root, dirs, files in os.walk(train_path, topdown=True):
    with open(train_label,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            tname = line.split('_')
            vid = tname[0]
            timg = [line] 
            if vid in tnames:
                tnames[vid] += timg
            else:
                tnames[vid] = timg
        pickle.dump(tnames,tnames_p)
        tnames_p.close()
        f.close()


def merge_label(d1_dict,d2_dict):
    """
    merge two datasets dict
    :param d1_dict: dataset1 dict
    :param d2_dict: dataset2 path 
    :return: merged dict
    """
    d1_id_cnt = len(d1_dict)
    d2_id_cnt = len(d2_dict)
    d1_items = d1_dict.items()
    d2_items = d2_dict.items()
    merge_dict = {}
    cnt = 0
    for key,value in d1_items:
        cnt += 1
        merge_dict[cnt] = value
    for key,value in d2_items:
        cnt += 1
        merge_dict[cnt] = value

    return merge_dict

def copy_ori2dst(ori_dict,ori_path_d1,ori_path_d2,save_path,d1_id_cnt,d2_id_cnt):
    """
    copy ori folder to destination folder
    :param ori_dict: original dict
    :param ori_path_d1: the original dataset1 path 
    :param ori_path_d2: the original dataset2 path 
    :param save_path: the final path which is going to be saved
    :return: none
    """
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    cnt = 1
    for item in ori_dict.items():
        tvid = str(item[0])
        timgs = item[1]
        # print(timgs)
        if cnt <= d1_id_cnt:
            for timg in timgs:
                src_path = ori_path_d1 + '/' + timg
                dst_path = save_path
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/'+tvid+'_'+timg)
        else:
#            embed()
            for timg in timgs:
                src_path = ori_path_d2 + '/' + timg
                dst_path = save_path
                timg = timg.split('_')[2]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                if os.path.exists(src_path):
                    copyfile(src_path, dst_path + '/'+tvid+'_'+timg+'.jpg')
                else:
                    continue
        cnt += 1

def ori2dst_split(ori_dict,ori_path,save_path):
    """
    copy ori folder to destination folder
    :param ori_dict: original dict
    :param ori_path: the original path 
    :param save_path: the final path which is going to be saved
    :return: none
    """
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for item in ori_dict.items():
        tvid = item[0]
        timgs = item[1]
        # print(timgs)
        for timg in timgs:
            src_path = ori_path + '/' + timg
            dst_path = save_path
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/'+tvid+'_'+timg)

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def _traceback(D):
    i,j = array(D.shape)-1
    p,q = [i],[j]
    while (i>0) or (j>0):
        tb = argmin((D[i,j-1], D[i-1,j]))
        if tb == 0:
            j -= 1
        else: #(tb==1)
            i -= 1
        p.insert(0,i)
        q.insert(0,j)
    return array(p), array(q)

def dtw(dist_mat):
    m, n = dist_mat.shape[:2]
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]
    path = _traceback(dist)
    return dist[-1,-1]/sum(dist.shape), dist, path

def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will Redo. Don't worry. Just chill".format(img_path))
            pass
    return img

def img_to_tensor(img,transform):
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def show_feature(x):
    for j in range(len(x)):
        for i in range(len(64)):
            ax = plt.subplot(4,16,i+1)
            ax.set_title('No #{}'.format(i))
            ax.axis('off')
            plt.imshow(x[j].cpu().data.numpy()[0,i,:,:],cmap='jet')
        plt.show()

def feat_flatten(feat):
    shp = feat.shape
    feat = feat.reshape(shp[0] * shp[1], shp[2])
    return feat

def show_similar(local_img_path, img_path, similarity, bbox):
    img1 = cv2.imread(local_img_path)
    img2 = cv2.imread(img_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (64, 128))
    img2 = cv2.resize(img2, (64, 128))
    cv2.rectangle(img1, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)

    p = np.where(similarity == np.max(similarity))
    y, x = p[0][0], p[1][0]
    cv2.rectangle(img2, (x - bbox[2] / 2, y - bbox[3] / 2), (x + bbox[2] / 2, y + bbox[3] / 2), (0, 255, 0), 1)
    plt.subplot(1, 3, 1).set_title('patch')
    plt.imshow(img1)
    plt.subplot(1, 3, 2).set_title(('max similarity: ' + str(np.max(similarity))))
    plt.imshow(img2)
    plt.subplot(1, 3, 3).set_title('similarity')
    plt.imshow(similarity)

def show_alignedreid(local_img_path, img_path, dist):
    def drow_line(img, similarity):
        for i in range(1, len(similarity)):
            cv2.line(img, (0, i*16), (63, i*16), color=(0,255,0))
            cv2.line(img, (96, i*16), (160, i*16), color=(0,255,0))
    def drow_path(img, path):
        for i in range(len(path[0])):
            cv2.line(img, (64, 8+16*path[0][i]), (96,8+16*path[1][i]), color=(255,255,0))
    img1 = cv2.imread(local_img_path)
    img2 = cv2.imread(img_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (64,128))
    img2 = cv2.resize(img2, (64,128))
    img = np.zeros((128,160,3)).astype(img1.dtype)
    img[:,:64,:] = img1
    img[:,-64:,:] = img2
    drow_line(img, dist)
    d,D,sp = dtw(dist)
    origin_dist = np.mean(np.diag(dist))
    drow_path(img, sp)
    plt.subplot(1,2,1).set_title('Aligned distance: %.4f \n Original distance: %.4f' %(d,origin_dist))
    plt.subplot(1,2,1).set_xlabel('Aligned Result')
    plt.imshow(img)
    plt.subplot(1,2,2).set_title('Distance Map')
    plt.subplot(1,2,2).set_xlabel('Right Image')
    plt.subplot(1,2,2).set_ylabel('Left Image')
    plt.imshow(dist)
    plt.subplots_adjust(bottom=0.1, left=0.075, right=0.85, top=0.9)
    cax = plt.axes([0.9, 0.25, 0.025, 0.5])
    plt.colorbar(cax = cax)
    plt.show()

def merge_feature(feature_list, shp, sample_rate = None):
    def pre_process(torch_feature_map):
        numpy_feature_map = torch_feature_map.cpu().data.numpy()[0]
        numpy_feature_map = numpy_feature_map.transpose(1,2,0)
        shp = numpy_feature_map.shape[:2]
        return numpy_feature_map, shp
    def resize_as(tfm, shp):
        nfm, shp2 = pre_process(tfm)
        scale = shp[0]/shp2[0]
        nfm1 = nfm.repeat(scale, axis = 0).repeat(scale, axis=1)
        return nfm1
    final_nfm = resize_as(feature_list[0], shp)
    for i in range(1, len(feature_list)):
        temp_nfm = resize_as(feature_list[i],shp)
        final_nfm = np.concatenate((final_nfm, temp_nfm),axis =-1)
    if sample_rate > 0:
        final_nfm = final_nfm[0:-1:sample_rate, 0:-1,sample_rate, :]
    return final_nfm
