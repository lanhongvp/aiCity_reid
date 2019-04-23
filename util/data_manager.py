# -*- coding: UTF-8 -*-
from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
from IPython import embed
# from util.utils import mkdir_if_missing, write_json, read_json

"""Image ReID"""

class aiCityVeRi(object):
    """
        AiCity_VeRi

        Reference:
        Liu et al. Large-scale vehicle re-identification in urban surveillance videos. ICME 2016.

        Dataset statistics:
        # vehicles: AICITY 300 (train) + 30 (val) + 333 (test)
                    VERI   575 (train) + 200 (test)
                    MERGE  875 (train) + 230 (val)
        # images: AICITY 32457 (train) + 4448 (t_gal) + 30 (t_val) + 18290 (test) + 1052(query)
                  VERI 37751 (train) + 11517 (test) + 1678 (query)

        """
    dataset_dir = 'aiCity_veri'

    def __init__(self, root='../../lan_reid/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = osp.abspath(self.dataset_dir)
        #embed()
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.train_dir_all = osp.join(self.dataset_dir,'image_train_all')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        # self.label_dir = self.dataset_dir + 'train_label.csv'
        self._check_before_run()

        train, num_train_vids, num_train_imgs = self._process_dir(self.train_dir,is_train=True,relabel=True)
        train_all,num_train_vids_all,num_train_imgs_all = self._process_dir(self.train_dir_all,is_train=True,relabel=True)
        query, num_query_vids, num_query_imgs = self._process_dir(self.query_dir,is_train=False)
        gallery, num_gallery_vids, num_gallery_imgs = self._process_dir(self.gallery_dir,is_train=False)
        num_total_pids = num_train_vids + num_query_vids
        #num_total_pids = 666
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> aiCity and VeRi datasets are loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_vids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_vids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_vids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.train_all = train_all
        self.query = query
        self.gallery = gallery

        self.num_train_vids = num_train_vids
        self.num_train_vids_all = num_train_vids_all
        self.num_query_vids = num_query_vids
        self.num_gallery_vids = num_gallery_vids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.train_dir_all):
            raise RuntimeError("'{}' is not available".format(self.train_dir_all))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def vid2label(self,id):
        sort_id = list(set(id))
        sort_id.sort()
        num_id = len(sort_id)
        label = []
        for x in id:
            label.append(sort_id.index(x))

        return label, num_id

    def _process_dir(self,dir_path, is_train=True, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        vid_list = []
        for img_path in img_paths:
            veid = int(img_path.split('/')[-1].split('_')[0])
            vid_list.append(veid)

        vlabel, num_vids = self.vid2label(vid_list)

        dataset = []
        count = 0
        for img_path in img_paths:
            vid = vid_list[count]
            if relabel:
                vid = vlabel[count]
            dataset.append((img_path, vid))
            count = count + 1
        #embed()
        num_imgs = len(dataset)
        return dataset, num_vids, num_imgs

    
class aiCityVeRi_t(object):
    """
        AiCity_test

        Reference:
        Liu et al. Large-scale vehicle re-identification in urban surveillance videos. ICME 2016.

        Dataset statistics:
        # vehicles: 333（train）+ 333（test）
        # images: 36953 (train) + 18290 (test) + 1052(query)
        """
    dataset_dir = 'aiCity'

    def __init__(self, root='../../lan_reid/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = osp.abspath(self.dataset_dir)
        #embed()
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        # self.label_dir = self.dataset_dir + 'train_label.csv'
        self._check_before_run()

        train, num_train_vids, num_train_imgs = self._process_dir(self.train_dir,is_train=True,relabel=True)
        query, num_query_vids, num_query_imgs = self._process_dir(self.query_dir,is_train=False)
        gallery, num_gallery_vids, num_gallery_imgs = self._process_dir(self.gallery_dir,is_train=False,is_track=True)
        # num_total_pids = num_train_vids + num_query_vids
        num_total_pids = 666
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> aiCityVeRi 666 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_vids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_vids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_vids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_vids = num_train_vids
        self.num_query_vids = num_query_vids
        self.num_gallery_vids = num_gallery_vids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def vid2label(self,id):
        sort_id = list(set(id))
        sort_id.sort()
        num_id = len(sort_id)
        label = []
        for x in id:
            label.append(sort_id.index(x))

        return label, num_id

    def _process_dir(self,dir_path,is_train=True,relabel=False,is_track=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        label_dir = self.dataset_dir + '/train_label.csv'
        test_track_dir = self.dataset_dir + '/test_track.txt'
        train_track_dir = self.dataset_dir + '/train_track.txt'

        vid_list = []
        if is_train and not is_track:
            train_label = open(label_dir)
            # cid_list = []
            img_tnames = []
            for line in train_label.readlines():
                vid = int(line.strip('.jpg').split(',')[0])
                img_tname = osp.join(dir_path,line.split(',')[1]).strip()
                # caid = int(img_path.split(osp.sep)[-1].split('.')[0].split('_')[1][1:])
                vid_list.append(vid)
                img_tnames.append(img_tname)
                # cid_list.append(caid)

            vlabel, num_vids = self.vid2label(vid_list)

            dataset = []
            count = 0
            for img_path in img_paths:
                #print('img_path',img_path)
                vid = vid_list[count]
                # cid = cid_list[count]
                if relabel:
                    vid = vlabel[count]
                dataset.append((img_tnames[count], vid))
#embed()
                count = count + 1
            # print('dataset\n',(dataset))
            num_imgs = len(dataset)
        elif not is_train and  not is_track :
            dataset = []
            count = 0
            for img_path in img_paths:
                dataset.append(img_path)
                count = count + 1
            num_imgs = len(dataset)
            num_vids = -1
            # print(dataset)
        elif not is_train and is_track:
            test_tracks = open(test_track_dir)
            dataset = []
        
            for test_track in test_tracks.readlines():
                test_track = test_track.split()
                test_track_ids = list(map(lambda track_id_:dir_path+'/'+track_id_,test_track))
                for t_id_dir in test_track_ids:
                    dataset.append(t_id_dir)
            # print('gallery track id',(dataset[0]))
            num_vids = -1
            num_imgs = len(dataset)
        return dataset, num_vids, num_imgs


class VeRi(object):
    """
        VeRi

        Reference:
        Liu et al. Large-scale vehicle re-identification in urban surveillance videos. ICME 2016.

        Dataset statistics:
        # vehicles: 576（train）+ 200（test）
        # images: 37778 (train) + 11579 (test) + 1678(query)
        """
    dataset_dir = 'VeRi'

    def __init__(self, root='../../data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        train, num_train_vids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_vids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_vids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_vids + num_query_vids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> VeRi-776 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_vids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_vids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_vids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_vids = num_train_vids
        self.num_query_vids = num_query_vids
        self.num_gallery_vids = num_gallery_vids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def vid2label(self,id):
        sortid = list(set(id))
        sortid.sort()
        numofid = len(sortid)
        label = []
        for x in id:
            label.append(sortid.index(x))

        return label, numofid

    def _process_dir(self,dir_path, relabel=False):
        print('dir path',dir_path)
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        vid_list = []
        cid_list = []
        for img_path in img_paths:
            #print('img_path',img_path)
            veid = int(img_path.split(osp.sep)[-1].split('.')[0].split('_')[0])
            caid = int(img_path.split(osp.sep)[-1].split('.')[0].split('_')[1][1:])
            vid_list.append(veid)
            cid_list.append(caid)

        vlabel, num_vids = self.vid2label(vid_list)

        dataset = []
        count = 0
        for img_path in img_paths:
            vid = vid_list[count]
            cid = cid_list[count]
            if relabel:
                vid = vlabel[count]
            dataset.append((img_path, vid, cid))
            #embed()
            count = count + 1

        num_imgs = len(dataset)
        return dataset, num_vids, num_imgs

class VehicleID(object):
    """
        VehicleID

        Reference:
        Liu et al. Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles. CVPR 2016.

        URL: https://pkuml.org/resources/pku-vehicleid.html

        Dataset statistics:
        # vehicles: 13134（train）+ 13133（test）
        # images: 110178 (train) + 111585 (test)
        """
    dataset_dir = 'VehicleID'

    def __init__(self, root='dataset', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        #self.train_dir = osp.join(self.dataset_dir, 'train')
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query_2400')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_2400')
        self.train_test_split = osp.join(self.dataset_dir, 'train_test_split')
        train_group = self.return_vid_vno(osp.join(self.train_test_split, 'train_list.txt'))
        test_800_group = self.return_vid_vno(osp.join(self.train_test_split, 'test_list_2400.txt'))

        self._check_before_run()

        #train, num_train_vids, num_train_imgs = self._process_dir(self.train_dir,train_group,relabel=True)
        train, num_train_vids, num_train_imgs = self._process_dir(self.train_dir,train_group, relabel=True)
        query, num_query_vids, num_query_imgs = self._process_dir(self.query_dir, test_800_group, relabel=False)
        gallery, num_gallery_vids, num_gallery_imgs = self._process_dir(self.gallery_dir,test_800_group, relabel=False)
        num_total_vids = num_train_vids + num_query_vids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> VehicleID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_vids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_vids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_vids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_vids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_vids = num_train_vids
        self.num_query_vids = num_query_vids
        self.num_gallery_vids = num_gallery_vids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
        if not osp.exists(self.test_800):
            raise RuntimeError("'{}' is not available".format(self.test_800))
        if not osp.exists(self.train_test_split):
            raise RuntimeError("'{}' is not available".format(self.train_test_split))

    def return_vid_vno(self,txt_path):
        f = open(txt_path)
        line = f.readline()
        d = {}
        while line:
            data = line.split()
            d[data[0]] = data[1]
            line = f.readline()
        f.close()

        return d

    def vid2label(self,id):
        sortid = list(set(id))
        sortid.sort()
        numofid = len(sortid)
        label = []
        for x in id:
            label.append(sortid.index(x))

        return label, numofid

    def _process_dir(self,dir_path, group, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        vid_list = []
        for img_path in img_paths:
            veid = int(group[img_path.split(osp.sep)[-1].split('.')[0]])
            vid_list.append(veid)

        vlabel, num_vids = self.vid2label(vid_list)

        dataset = []
        count = 0
        for img_path in img_paths:
            vid = vid_list[count]
            if relabel:
                vid = vlabel[count]
            dataset.append((img_path, vid))
            count = count + 1

        num_imgs = len(dataset)
        return dataset, num_vids, num_imgs


"""Create dataset"""

__img_factory = {
    'VehicleID': VehicleID,
    'aiCityVeRi':aiCityVeRi,
    'VeRi':VeRi,
    'aiCityTest':aiCityVeRi_t,
}

def get_names():
    return list(__img_factory.keys())

def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)


#if __name__ == '__main__':
#   aiCityVeRi()
