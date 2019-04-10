import os
from shutil import copyfile
import pickle
from util.utils import *

# You only need to change this line to your dataset download path
download_path = '../data_b/aiCity'
save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# write the train data to pickle
write_pickle(download_path)
f = open('tnames.pkl','rb')
tnames = pickle.load(f)    # all train data info
f.close()

# obtain the gallery/query/train part
tnames_gallery = dict_key_slice(tnames,0,30)
print('gallery id',len(tnames_gallery))
tnames_query = dict_value_slice(tnames_gallery,0,1)
print('query id',len(tnames_query))
tnames_train = dict_key_slice(tnames,30,-1)
print('train id',len(tnames_train))

# save the train_all/gallery/query/train part to corresponding folder
#---------------------------------------
#train_all
train_path = download_path + '/image_train'
ta_save_path = download_path + '/pytorch/train_all'
copy_ori2dst(tnames,train_path,ta_save_path)

#---------------------------------------
#train_query_gallery
train_save_path = download_path + '/pytorch/train'
query_save_path = download_path + '/pytorch/query'
gallery_save_path = download_path + '/pytorch/gallery'

copy_ori2dst(tnames_train,train_path,train_save_path)
copy_ori2dst(tnames_query,train_path,query_save_path)
copy_ori2dst(tnames_gallery,train_path,gallery_save_path)




