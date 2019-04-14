# AI_CITY
## 实验结果记录
- [实验记录](https://github.com/lanhongvp/aiCity_reid/blob/master/exp.md)
## 框架介绍
```
|--aiCity_reid             
    |--aligned 
        |--HorizontalMaxPool2D.py [local部分的max_pooling]
        |--local_dist.py [triplet loss调用的距离计算脚本]

    |--util [常用脚本汇总]
        |--data_manager.py [源数据处理]
        |--dataset_loader.py [结合pytorch，生成dataloader]
        |--distance.py [距离计算]
        |--eval_metrics.py [评价矩阵，计算mAP,CMC]
        |--FeatureExtractor.py [pytorch特征提取]
        |--losses.py [pytorch常用的计算loss]
        |--optimizers.py [pytorch常用的优化器类]
        |--re_ranking.py [测试时，对query的结果进行重排序]
        |--samplers.py [随机ID选取]
        |--test_CMC.py [test_aicity.py调用**test_rank100_aicity**，生成对应的测试结果]
        |--transforms.py [数据增强的操作]
        |--utils.py [辅助函数]
    
    |--models [常用模型]
        |--DenseNet.py 
        |--InceptionV4.py 
        |--localDenseNet.py
        |--PCB.py [目前采用**class DensePCB(nn.module)**]
        |--ResNet.py
        |--ShuffleNet.py
    
    |--test_aicity.py [测试脚本，网络前传，按照比赛要求输出测试结果]  
    |--train_aicity.py [训练脚本，主函数，模型训练]
    |--prepare_data.py [源数据处理]
    |--test.sh [测试运行脚本]
    |--train.sh [训练运行脚本]
```
## 框架简单使用
### 数据准备
### 1 python prepare_data.py
- 此脚本将原始数据划分为train/val/train_all
- 将原图片`imgname.jpg`重命名为`vid_imgname.jpg`
- 运行脚本，处理原始数据(默认路径`data/aiCity`)，将处理之后的数据存放于`data/aiCity_s`
    - image_train
    - image_test
    - image_val
    - image_train_all
#### P.S
原来采用`VeRi`数据集进行验证，现在采用`prepare.py`处理`aiCity`的数据集，运用`aiCity`数据集划分的`image_val`进行验证
### 2 sh train.sh
模型训练脚本
### 3 sh test.sh
模型测试脚本
