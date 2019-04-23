import os
from shutil import copyfile
import pickle
from util.utils import *

# You only need to change this line to your dataset download path
download_path_aicity = '../data/aiCity'
download_path_veri = '../data/VeRi'
train_path_veri = download_path_veri + '/name_train.txt'
gallery_path_veri = download_path_veri + '/name_test.txt'
query_path_veri = download_path_veri + '/name_query.txt'

save_path = '../data/aiCity_veri'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

# write the train data to pickle
write_pickle_aicity(download_path_aicity)
write_pickle_veri(train_path_veri,'veri_train')
write_pickle_veri(gallery_path_veri,'veri_gallery')
write_pickle_veri(query_path_veri,'veri_query')

f_aicity_t = open('aicity_train.pkl','rb')
f_veri_t = open('veri_train.pkl','rb')
f_veri_q = open('veri_query.pkl','rb')
f_veri_g = open('veri_gallery.pkl','rb')

# get dict value for two datasets
aicity_trian = pickle.load(f_aicity_t)    # all train data info
veri_train = pickle.load(f_veri_t)
veri_gallery = pickle.load(f_veri_g)
veri_query = pickle.load(f_veri_q)

print('aicity_train',len(aicity_trian))
print('veri_train',len(veri_train))
print('veri_gallery',len(veri_gallery))
print('veri_query',len(veri_query))

f_aicity_t.close()
f_veri_t.close()
f_veri_g.close()
f_veri_q.close()

# get aicity gallery and query part from the train dataset
print("AICITY PART\n")
t_gallery_aicity = dict_key_slice(aicity_trian,0,30)
t_gallery_aicity = dict_value_slice(t_gallery_aicity,1,-1)
print('t_gallery_id ',len(t_gallery_aicity))
t_query_aicity = dict_value_slice(t_gallery_aicity,0,1)
print('t_query_id',len(t_query_aicity))
t_train_aicity = dict_key_slice(aicity_trian,30,-1)
print('t train id',len(t_train_aicity))
print('train all id',len(aicity_trian))

#---------------------------------------
# train_all
train_img_aicity = download_path_aicity + '/image_train'
train_img_veri = download_path_veri + '/image_train'
ta_save_path = save_path + '/image_train_all'
train_merge_all = merge_label(aicity_trian,veri_train)
train_merge_part = merge_label(t_train_aicity,veri_train)
query_merge_all = merge_label(t_query_aicity,veri_query)
gallery_merge_all = merge_label(t_gallery_aicity,veri_gallery)

print('MERGE PART\n')
print('MERGE ALL TRAIN ID ',len(train_merge_all))
print('MERGE PART TRAIN ID ',len(train_merge_part))
print('MERGE QUERY ID ',len(query_merge_all))
print('MERGE GALLERY ID ',len(gallery_merge_all))

# train all merge
# AICITY:333 ID  VERI: 575 ID
copy_ori2dst(train_merge_all,train_img_aicity,train_img_veri,ta_save_path,len(aicity_trian),len(veri_train))

# #---------------------------------------
# train_query_gallery
query_img_veri = download_path_veri + '/image_query'
gallery_img_veri = download_path_veri + '/image_test'
train_save_path = save_path + '/image_train'
query_save_path = save_path + '/image_query'
gallery_save_path = save_path + '/image_test'

# train part merge
# AICITY:300 ID VERI: 575 ID
copy_ori2dst(train_merge_part,train_img_aicity,train_img_veri,train_save_path,len(t_train_aicity),len(veri_train))

# query merge
# AICITY: 33 ID VERI: 200 ID
copy_ori2dst(query_merge_all,train_img_aicity,query_img_veri,query_save_path,len(t_query_aicity),len(veri_query))

# gallery merge
copy_ori2dst(gallery_merge_all,train_img_aicity,gallery_img_veri,gallery_save_path,len(t_gallery_aicity),len(veri_gallery))





