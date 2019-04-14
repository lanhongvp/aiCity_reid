import os
from shutil import copyfile
import pickle
from util.utils import *

# You only need to change this line to your dataset download path
download_path = '../data/aiCity'
save_path = '../data/aiCity_sb'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# write the train data to pickle
write_pickle(download_path)
f = open('tnames.pkl','rb')
tnames = pickle.load(f)    # all train data info
#print('tnames',len(tnames))
f.close()

# obtain the gallery/query/train part
tnames_gallery = dict_key_slice(tnames,0,30)
tnames_gallery = dict_value_slice(tnames_gallery,1,-1)
print('gallery id',len(tnames_gallery))
tnames_query = dict_value_slice(tnames_gallery,0,1)
print('query id',len(tnames_query))
tnames_train = dict_key_slice(tnames,30,-1)
print('train id',len(tnames_train))
print('train all id',len(tnames))
# save the train_all/gallery/query/train part to corresponding folder
#---------------------------------------
#train_all
train_path = download_path + '/image_train'
ta_save_path = save_path + '/image_train_all'
#copy_ori2dst(tnames,train_path,ta_save_path)

#---------------------------------------
#train_query_gallery
train_save_path = save_path + '/image_train'
query_save_path = save_path + '/image_query'
gallery_save_path = save_path + '/image_test'

ori2dst_split(tnames_train,train_path,train_save_path)
ori2dst_split(tnames,train_path,ta_save_path)
ori2dst_split(tnames_query,train_path,query_save_path)
ori2dst_split(tnames_gallery,train_path,gallery_save_path)




