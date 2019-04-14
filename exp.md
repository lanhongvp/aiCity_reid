# 实验记录
## 0.exp0--baseline
### 0.1 backbone
- resnet50
### 0.2 loss
- softmax loss
### 0.3 feature
- global feature
### 0.4 初始学习率/学习率调整步骤
- lr = 0.0002
- 40epoch 未变
### 0.5 VeRi验证结果[epoch 35]
- mAP: 13.5%
- rank1: 40.3%
- rank5: 56.6%
- rank10: 66.8%
- rank20: 75.3%
### 0.6 aiCity验证结果
- mAP: 0.1714
- cmc1: 0.3905
- cmc5: 0.5048
- cmc10: 0.5638
- cmc15: 0.6133
- cmc20: 0.6533
- cmc30: 0.7010
- cmc100: 0.7486
## [1]调参系列实验
### 1.1 exp1_1--参数调整(学习率)
#### 1.1.1 backbone
- resnet50
#### 1.1.2 loss
- softmax loss
#### 1.1.3 feature
- global feature
#### 1.1.4 初始学习率/学习率调整步骤
- lr = 0.0002
- 20 epoch /40 epoch
#### 1.1.5 VeRi验证结果[epoch: 5]
- mAP: 14.9%
- rank1: 43.1%
- rank5: 61.1%
- ran10: 70.1%
- rank20: 76.8%
### 1.2 exp1_2--参数调整(学习率)
#### 1.2.1 backbone
- resnet50
#### 1.2.2 loss
- softmax loss
#### 1.2.3 feature
- global feature
#### 1.2.4 初始学习率/学习率调整步骤
- lr = 0.0003
- 20 epoch/ 40 epoch
#### 1.2.5 VeRi验证结果
- mAP: 12.4%
- rank1: 38.4%
- rank5: 54.3%
- rank10: 63.5%
- rank20: 72.2%
## [2] re-ranking系列实验
### 2.1 global-feature -- re-ranking
#### 2.1.1 backbone
- resnet50
#### 2.1.2 loss
- softmax loss
#### 2.1.3 feature
- global feature
#### 2.1.4 初始学习率/调整步骤
- 同1.1 exp1_1
#### 2.1.5 aiCity验证结果
- mAP: 0.2134
- cmc1: 0.3771
- cmc5: 0.4171
- cmc10: 0.4476
- cmc15: 0.4876
- cmc20: 0.5124
- cmc30: 0.5771
- cmc100: 0.6800
### 2.2 global-feature -- re-ranking + test_track
#### 2.2.1 backbone
- resnet50
#### 2.2.2 loss
- softmax loss
#### 2.2.3 feature
- global feature
#### 2.2.4 初始学习率/调整步骤
- 同1.1 exp1_1
#### 2.2.5 aiCity验证结果
- mAP: 0.2535 
- cmc1: 0.3733
- cmc5: 0.3733
- cmc10: 0.3790
- cmc15: 0.3962
- cmc20: 0.4286
- cmc30: 0.4781
- cmc100: 0.5410
### 3.1[PCB] local-feature -- re-ranking + test_track
#### 3.1.1 backbone
- resnet50
#### 3.1.2 loss
- softmax loss
#### 3.1.3 feature
- local feature
- 384 X 128
#### 3.1.4 初始学习率/调整步骤
- 同1.1 exp1_1(lr=0.0002)
#### 3.1.5 aiCity验证结果
- mAP: 0.2984 
- cmc1: 0.4571
- cmc5: 0.4571
- cmc10: 0.4590
- cmc15: 0.4705
- cmc20: 0.4857
- cmc30: 0.5390
- cmc100: 0.5924
#### 3.1.6 veRi验证结果
- mAP: 0.172
- rank1: 0.507
- rank5: 0.649
- rank10: 0.723
- rank20: 0.797
### 3.2[PCB] local-feature -- re-ranking + test_track
#### 3.2.1 backbone
- densenet121
- **384 X 128**
#### 3.2.2 loss
- softmax loss
#### 3.2.3 feature
- local feature
#### 3.2.4 初始学习率/调整步骤
- 同1.1 exp1_1(lr=0.0002)
#### 3.2.5 aiCity验证结果(未resize)
- mAP: 0.2791
- cmc1: 0.4210
- cmc5: 0.4210
- cmc10: 0.4248
- cmc15: 0.4552
- cmc20: 0.4762
- cmc30: 0.5105
- cmc100: 0.5695
#### 3.2.6.1 veRi验证结果(lr=0.0002)
- mAP: 0.174
- rank1: 0.516
- rank5: 0.670
- rank10: 0.740
- rank20: 0.806
#### 3.2.6.2 veRi验证结果(lr=0.0003)
- mAP: 0.162
- rank1: 0.484
- rank5: 0.620
- rank10: 0.708
- rank20: 0.785
### 3.2b[PCB] local feature -- re-ranking + test_track
#### 3.2b.1 backbone
- densenet121
- **256 X 256**
#### 3.2.2b loss
- softmax loss
#### 3.2.3b feature
- local feature
#### 3.2.4b 初始学习率/调整步骤
- 同1.1 exp1_1(lr=0.0002)
- 同1.2 exp1_2(lr=0.0003)
#### 3.2.5.1b aiCity验证结果(le=0.0002/256X256)
- mAP: 0.3030
- cmc1: 0.4590
- cmc5: 0.4590
- cmc10: 0.4648
- cmc15: 0.4876
- cmc20: 0.5143
- cmc30: 0.5619
- cmc100: 0.6248
#### 3.2.5.2b aiCity验证结果(lr=0.0002/256X256/epoch=60/without rerank,track)
- mAP: 0.2423
- cmc1: 0.3143
- cmc5: 0.3143
- cmc10: 0.3276
- cmc15: 0.3410
- cmc20: 0.3810
- cmc30: 0.4438
- cmc100: 0.5067
#### 3.2.6.1b veRi验证结果(lr=0.0002)--log_small_lr_2
- mAP: 0.189
- rank1: 0.527
- rank5: 0.674
- rank10: 0.738
- rank20: 0.812
#### 3.2.6.2b veRi验证结果(lr=0.0003)--log_2
- mAP: 0.171
- rank1: 0.527
- rank5: 0.664
- rank10: 0.730
- rank20: 0.803
### 3.3[PCB] lf+gf -- re-ranking + test_track
#### 3.3.1 backbone
- densenet121
- **256 X 256**
#### 3.3.2 loss
- softmax loss
#### 3.3.3 feature
- local feature
- global feature
#### 3.3.4 初始学习率/调整步骤
- 同1.1 exp1_1(lr=0.0002)
- 同1.2 exp1_2(lr=0.0003)
#### 3.3.5.1 aiCity验证结果(veRi=best_model)
- mAP: 0.3001 
- cmc1: 0.4152 
- cmc5: 0.4152
- cmc10: 0.4267
- cmc15: 0.4514
- cmc20: 0.4781
- cmc30: 0.5276
- cmc100: 0.6114
#### 3.3.5.2 aiCity验证结果(veRi=epoch60)
- mAP: 0.3071
- cmc1: 0.3790 
- cmc5: 0.3790
- cmc10: 0.3924
- cmc15: 0.4057
- cmc20: 0.4362
- cmc30: 0.4800
- cmc100: 0.5371
#### 3.3.6.1 veRi验证结果(lr=0.0002)
- mAP: 0.208
- rank1: 0.575
- rank5: 0.711
- rank10: 0.781
- rank20: 0.852
#### 3.3.6.2 veRi验证结果(lr=0.0003)
- mAP: 0.216
- rank1: 0.584
- rank5: 0.711
- rank10: 0.789
- rank20: 0.851
### 4.1 网络修改实验 global_feature --reranking + test_track
#### 4.1.1 backbone
- densenet121
#### 4.1.2 loss
- softmax loss
#### 4.1.3 backbone
- densenet121
#### 4.1.4 初始学习率/调整步骤
- 同1.2 exp1_2(lr=0.0003)
- 同1.1 exp1_1(lr=0.0002)
#### 4.1.5 aiCity验证结果 
- mAP: 0.3011
- cmc1: 0.4438
- cmc5: 0.4438
- cmc10: 0.4457
- cmc15: 0.4705
- cmc20: 0.4952
- cmc30: 0.5429
- cmc100: 0.5981
#### 4.1.6 veRi验证结果(lr=0.0003)
- mAP: 0.209
- rank1: 0.605
- rank5: 0.737
- rank10: 0.792
- rank20: 0.845
#### 4.1.7 veRi验证结果(lr=0.0002)
- mAP: 0.199
- rank1: 0.597
- rank5: 0.735
- rank10: 0.793
- rank20: 0.858
### 4.2 数据集划分验证实验 gf+lf --reranking + test_track
#### 4.2.1 backbone
- densenet121
#### 4.2.2 loss
- softmax loss
#### 4.2.3 feature
- gf
- lf
#### 4.2.4 初始学习率/调整步骤
- 同1.2 exp1_2(lr=0.0003)
- 同1.1 exp1_1(lr=0.0002)
#### 4.2.5.1 aiCity验证结果(30ID验证,only gfi,epoch=25) 
- mAP: 0.781
- cmc1: 1.00
- cmc5: 1.00
- cmc10: 1.00
- cmc20: 1.00
#### 4.2.5.2 aiCity验证结果(30ID验证,gf and lf,epoch=10) 
- mAP: 0.792
- cmc1: 1.00
- cmc5: 1.00
- cmc10: 1.00
- cmc20: 1.00
### 5.1 [PCB]triplet loss实验 gf+lf --reranking + test_track
#### 5.1.1 backbone
- densenet121
- pcb
#### 5.1.2 loss
- triplet loss
- softmax loss
#### 5.1.3 feature
- gf
- lf
- dropout(0.5)
#### 5.1.4 初始学习率/调整步骤
- 同 exp1_1(lr=0.0002)
#### 5.1.5 aiCity验证结果(30ID验证) 
- mAP: 0.3011
- cmc1: 0.4438
- cmc5: 0.4438
- cmc10: 0.4457
- cmc20: 0.4705
### 5.2 triplet loss实验 gf+lf --reranking + test_track
#### 5.1.1 backbone
- densenet121
- pcb
#### 5.1.2 loss
- triplet loss
- softmax loss
#### 5.1.3 feature
- gf
- lf
#### 5.1.4 初始学习率/调整步骤
- 同1.2 exp1_2(lr=0.0003)
- 同1.1 exp1_1(lr=0.0002)
#### 5.1.5 aiCity验证结果(30ID验证,only gf) 
- mAP: 0.3011
- cmc1: 0.4438
- cmc5: 0.4438
- cmc10: 0.4457
- cmc20: 0.4705
