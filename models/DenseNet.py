from __future__ import absolute_import

import torch
from torch.nn import init
from torch import nn
from torch.nn import functional as F
import torchvision
from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d
from IPython import embed
from collections import OrderedDict
from torchvision.models.densenet import _DenseBlock,_Transition
import numpy as np

__all__ = ['DenseNet121', 'MyDenseNet_stn','MyDenseNet_stn_local']

class DenseNet121(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False,**kwargs):
        super(DenseNet121, self).__init__()
        self.loss = loss
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.classifier = nn.Linear(1024, num_classes)
        self.feat_dim = 1024 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(1024)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f, lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class DenseNet_Alignment(nn.Module):
    def __init__(self, model_path,growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,**kwargs):

        super(DenseNet_Alignment, self).__init__()

        # First convolution
        self.firstconvolution = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.features = nn.Sequential()
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f
        out = self.classifier(f)
        return out

class MyDenseNet_stn(nn.Module):
    def __init__(self, model_path,growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,**kwargs):

        super(MyDenseNet_stn, self).__init__()

        #Branch1 First convolution
        self.firstconvolution = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Branch1 denseblock 1
        DBnum_features = num_init_features
        TRnum_features = DBnum_features + block_config[0]*growth_rate
        self.block1 = nn.Sequential(OrderedDict([
            ('denseblock1', _DenseBlock(num_layers=block_config[0], num_input_features=DBnum_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
            ('transition1', _Transition(num_input_features=TRnum_features, num_output_features=TRnum_features // 2))]
        ))

        #Branch1 denseblock 2
        DBnum_features = TRnum_features // 2
        TRnum_features = DBnum_features + block_config[1] * growth_rate
        self.block2 = nn.Sequential(OrderedDict([
            ('denseblock2', _DenseBlock(num_layers=block_config[1], num_input_features=DBnum_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
            ('transition2', _Transition(num_input_features=TRnum_features, num_output_features=TRnum_features // 2))]
        ))

        #Branch1 denseblock 3
        DBnum_features = TRnum_features // 2
        TRnum_features = DBnum_features + block_config[2] * growth_rate
        self.highnum_fea = DBnum_features
        self.trannum_fea = TRnum_features
        self.block3 = nn.Sequential(OrderedDict([
            ('denseblock3', _DenseBlock(num_layers=block_config[2], num_input_features=DBnum_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
            ('transition3', _Transition(num_input_features=TRnum_features, num_output_features=TRnum_features // 2))]
        ))

        #Branch1 denseblock 4
        DBnum_features = TRnum_features // 2
        TRnum_features = DBnum_features + block_config[3] * growth_rate
        self.block4 = nn.Sequential(OrderedDict([
            ('denseblock4', _DenseBlock(num_layers=block_config[3], num_input_features=DBnum_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate))]
        ))


        #Branch1 Final batch norm
        self.norm5 = nn.BatchNorm2d(TRnum_features)

        #Branch1 Linear layer
        self.classifier = nn.Linear(TRnum_features, num_classes)

        #Spatical transformer Localization network
        self.localization = nn.Sequential(OrderedDict([
        ('denseblock3', _DenseBlock(num_layers=24, num_input_features=self.highnum_fea,
                         bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
        ('transition3', _Transition(num_input_features=self.trannum_fea, num_output_features=self.trannum_fea // 2))]
        ))

        self.downsample = nn.Sequential(OrderedDict([
            ('bn', nn.BatchNorm2d(512)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(512,128,kernel_size=1,stride=1,padding=0,bias=True))
        ]))
        #Regressor
        self.fc_loc = nn.Sequential(
            nn.Linear(128*8*8,128*8),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32,3*2)
        )

        self.fc_loc[6].weight.data.fill_(0)
        self.fc_loc[6].bias.data = torch.FloatTensor([1,0,0,0,1,0])

        alignmodel = DenseNet_Alignment()
        self.alignbase = alignmodel.features
        self.alignclassifier = nn.Linear(TRnum_features, num_classes)

        self.load_param()

    def load_param(self):
        param_dict = torch.load('/workspace/densenet121-a639ec97.pth')
        standard_dict = {}
        temp_dict = {}

        for i in param_dict:
            if 'features' in i:
                key = i.replace('features.','',1)
            else:
                key = i
            temp_dict[key] = param_dict[i]

        for i in temp_dict:
            if 'norm.1' in i:
                key = i.replace('norm.1','norm1',1)
            elif 'conv.1' in i:
                key = i.replace('conv.1','conv1',1)
            elif 'norm.2' in i:
                key = i.replace('norm.2','norm2',1)
            elif 'conv.2' in i:
                key = i.replace('conv.2','conv2',1)
            else:
                key = i
            standard_dict[key] = temp_dict[i]

        for i in self.state_dict():
            if 'firstconvolution' in i:
                paramkey = i.replace('firstconvolution.','',1)
            elif 'localization' in i:
                paramkey = i.replace('localization.','',1)
            elif 'alignbase' in i:
                paramkey = i.replace('alignbase.','',1)
            elif 'block1' in i:
                paramkey = i.replace('block1.','',1)
            elif 'block2' in i:
                paramkey = i.replace('block2.','',1)
            elif 'block3' in i:
                paramkey = i.replace('block3.','',1)
            elif 'block4' in i:
                paramkey = i.replace('block4.','',1)
            elif 'classifier' in i or 'fc_loc' in i or 'alignclassifier' or 'downsample' in i:
                continue
            else:
                paramkey=i

            self.state_dict()[i].copy_(standard_dict[paramkey])

    def stn(self,x1,x2):
        xs = self.localization(x2)
        xs = self.downsample(xs)
        xs  =xs.view(-1,128*8*8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)

        grid = F.affine_grid(theta, x1.size())
        x = F.grid_sample(x1,grid)

        return x

    def forward(self, x):
        x = self.firstconvolution(x)
        featuremap1 = x
        x = self.block1(x)
        x = self.block2(x)
        featuremap2  = x
        x = self.block3(x)
        x = self.block4(x)
        x = self.norm5(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        normf = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        out = self.classifier(f)

        aligninput = self.stn(featuremap1,featuremap2)
        ax = self.alignbase(aligninput)
        ax = F.avg_pool2d(ax,ax.size()[2:])
        af = ax.view(ax.size(0),-1)
        normaf = 1. * af / (torch.norm(af, 2, dim=-1, keepdim=True).expand_as(af) + 1e-12)
        aout = self.alignclassifier(af)
        if not self.training:
            return torch.cat((normf,normaf),1)

        return (out,aout)

class MyDenseNet_stn_local(nn.Module):
    def __init__(self, model_path,growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,gh =False,ah=False,**kwargs):
        super(MyDenseNet_stn_local, self).__init__()
        self.gh = gh
        self.ah = ah
        #Branch1 First convolution
        self.firstconvolution = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Branch1 denseblock 1
        DBnum_features = num_init_features
        TRnum_features = DBnum_features + block_config[0]*growth_rate
        self.block1 = nn.Sequential(OrderedDict([
            ('denseblock1', _DenseBlock(num_layers=block_config[0], num_input_features=DBnum_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
            ('transition1', _Transition(num_input_features=TRnum_features, num_output_features=TRnum_features // 2))]
        ))

        #Branch1 denseblock 2
        DBnum_features = TRnum_features // 2
        TRnum_features = DBnum_features + block_config[1] * growth_rate
        self.block2 = nn.Sequential(OrderedDict([
            ('denseblock2', _DenseBlock(num_layers=block_config[1], num_input_features=DBnum_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
            ('transition2', _Transition(num_input_features=TRnum_features, num_output_features=TRnum_features // 2))]
        ))

        #Branch1 denseblock 3
        DBnum_features = TRnum_features // 2
        TRnum_features = DBnum_features + block_config[2] * growth_rate
        self.highnum_fea = DBnum_features
        self.trannum_fea = TRnum_features
        self.block3 = nn.Sequential(OrderedDict([
            ('denseblock3', _DenseBlock(num_layers=block_config[2], num_input_features=DBnum_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
            ('transition3', _Transition(num_input_features=TRnum_features, num_output_features=TRnum_features // 2))]
        ))

        #Branch1 denseblock 4
        DBnum_features = TRnum_features // 2
        TRnum_features = DBnum_features + block_config[3] * growth_rate
        self.block4 = nn.Sequential(OrderedDict([
            ('denseblock4', _DenseBlock(num_layers=block_config[3], num_input_features=DBnum_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate))]
        ))

        #Branch1 Final batch norm
        self.norm5 = nn.BatchNorm2d(TRnum_features)

        #Branch1 Linear layer
        self.classifier = nn.Linear(TRnum_features, num_classes)

        if self.gh:
            # =====================append conv for Horizontal======================= #
            self.localgh_conv = nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False)
            init.kaiming_normal(self.localgh_conv.weight, mode='fan_out')
            self.ghfeat_bn2d = nn.BatchNorm2d(256)
            init.constant(self.ghfeat_bn2d.weight, 1)
            init.constant(self.ghfeat_bn2d.bias, 0)

            ##------------------------------stripe----------------------------------------##
            self.ghinstance0 = nn.Linear(256, self.num_classes)
            init.normal(self.ghinstance0.weight, std=0.001)
            init.constant(self.ghinstance0.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.ghinstance1 = nn.Linear(256, self.num_classes)
            init.normal(self.ghinstance1.weight, std=0.001)
            init.constant(self.ghinstance1.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.ghinstance2 = nn.Linear(256, self.num_classes)
            init.normal(self.ghinstance2.weight, std=0.001)
            init.constant(self.ghinstance2.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.ghinstance3 = nn.Linear(256, self.num_classes)
            init.normal(self.ghinstance3.weight, std=0.001)
            init.constant(self.ghinstance3.bias, 0)
            ##------------------------------stripe----------------------------------------##

            self.ghdrop = nn.Dropout(0.5)

        #Spatical transformer Localization network
        self.localization = nn.Sequential(OrderedDict([
        ('denseblock3', _DenseBlock(num_layers=24, num_input_features=self.highnum_fea,
                         bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
        ('transition3', _Transition(num_input_features=self.trannum_fea, num_output_features=self.trannum_fea // 2))]
        ))

        self.downsample = nn.Sequential(OrderedDict([
            ('bn', nn.BatchNorm2d(512)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(512,128,kernel_size=1,stride=1,padding=0,bias=True))
        ]))
        #Regressor
        self.fc_loc = nn.Sequential(
            nn.Linear(128*8*8,128*8),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32,3*2)
        )

        self.fc_loc[6].weight.data.fill_(0)
        self.fc_loc[6].bias.data = torch.FloatTensor([1,0,0,0,1,0])

        alignmodel = DenseNet_Alignment()
        self.alignbase = alignmodel.features
        self.alignclassifier = nn.Linear(TRnum_features, num_classes)

        if self.ah:
            # =====================append conv for Horizontal======================= #
            self.localah_conv = nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False)
            init.kaiming_normal(self.localah_conv.weight, mode='fan_out')
            self.ahfeat_bn2d = nn.BatchNorm2d(256)
            init.constant(self.ahfeat_bn2d.weight, 1)
            init.constant(self.ahfeat_bn2d.bias, 0)

            ##------------------------------stripe----------------------------------------##
            self.ahinstance0 = nn.Linear(256, self.num_classes)
            init.normal(self.ahinstance0.weight, std=0.001)
            init.constant(self.ahinstance0.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.ahinstance1 = nn.Linear(256, self.num_classes)
            init.normal(self.ahinstance1.weight, std=0.001)
            init.constant(self.ahinstance1.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.ahinstance2 = nn.Linear(256, self.num_classes)
            init.normal(self.ahinstance2.weight, std=0.001)
            init.constant(self.ahinstance2.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.ahinstance3 = nn.Linear(256, self.num_classes)
            init.normal(self.ahinstance3.weight, std=0.001)
            init.constant(self.ahinstance3.bias, 0)
            ##------------------------------stripe----------------------------------------##

            self.ahdrop = nn.Dropout(0.5)

        self.load_param()

    def load_param(self):
        param_dict = torch.load('/workspace/densenet121-a639ec97.pth')
        standard_dict = {}
        temp_dict = {}

        for i in param_dict:
            if 'features' in i:
                key = i.replace('features.','',1)
            else:
                key = i
            temp_dict[key] = param_dict[i]

        for i in temp_dict:
            if 'norm.1' in i:
                key = i.replace('norm.1','norm1',1)
            elif 'conv.1' in i:
                key = i.replace('conv.1','conv1',1)
            elif 'norm.2' in i:
                key = i.replace('norm.2','norm2',1)
            elif 'conv.2' in i:
                key = i.replace('conv.2','conv2',1)
            else:
                key = i
            standard_dict[key] = temp_dict[i]

        for i in self.state_dict():
            if 'firstconvolution' in i:
                paramkey = i.replace('firstconvolution.','',1)
            elif 'localization' in i:
                paramkey = i.replace('localization.','',1)
            elif 'alignbase' in i:
                paramkey = i.replace('alignbase.','',1)
            elif 'block1' in i:
                paramkey = i.replace('block1.','',1)
            elif 'block2' in i:
                paramkey = i.replace('block2.','',1)
            elif 'block3' in i:
                paramkey = i.replace('block3.','',1)
            elif 'block4' in i:
                paramkey = i.replace('block4.','',1)
            elif 'classifier' in i or 'fc_loc' in i or 'alignclassifier' or 'downsample' in i:
                continue
            elif 'localgh_conv' in i or 'ghfeat_bn2d' in i or 'ghinstance0' or 'ghinstance1' in i or 'ghinstance2' in i\
                or 'ghinstance3' in i or 'ghdrop' in i:
                continue
            elif 'localah_conv' in i or 'ahfeat_bn2d' in i or 'ahinstance0' or 'ahinstance1' in i or 'ahinstance2' in i\
                or 'ahinstance3' in i or 'ahdrop' in i:
                continue
            else:
                paramkey=i

            self.state_dict()[i].copy_(standard_dict[paramkey])

    def stn(self,x1,x2):
        xs = self.localization(x2)
        xs = self.downsample(xs)
        xs  =xs.view(-1,128*8*8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)

        grid = F.affine_grid(theta, x1.size())
        x = F.grid_sample(x1,grid)

        return x

    def forward(self, x):
        x = self.firstconvolution(x)
        featuremap1 = x
        x = self.block1(x)
        x = self.block2(x)
        featuremap2  = x
        x = self.block3(x)
        x = self.block4(x)
        x = self.norm5(x)
        if self.gh:
            ghx = x
            sx = ghx.size(2)/4
            kx = ghx.size(2) -sx*3
            ghx = F.avg_pool2d(ghx, kernel_size=(kx,ghx.size(3)),stride=(sx,ghx.size(3)))

            ghx = self.ghdrop(ghx)
            ghx = self.localgh_conv(ghx)
            ghx = self.ghfeat_bn2d(ghx)
            ghx = F.relu(ghx)

            ghx = ghx.chunk(4,2)
            ghx0 = ghx[0].contiguous().view(ghx[0].size(0),-1)
            ghx1 = ghx[1].contiguous().view(ghx[1].size(0), -1)
            ghx2 = ghx[2].contiguous().view(ghx[2].size(0), -1)
            ghx3 = ghx[3].contiguous().view(ghx[3].size(0),-1)
            ghfeature = torch.cat((ghx0,ghx1,ghx2,ghx3),1)
            ghnorm = 1. * ghfeature / (torch.norm(ghfeature, 2, dim=-1, keepdim=True).expand_as(ghfeature) + 1e-12)
            ghc0 = self.ghinstance0(ghx0)
            ghc1 = self.ghinstance1(ghx1)
            ghc2 = self.ghinstance2(ghx2)
            ghc3 = self.ghinstance3(ghx3)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        normf = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        out = self.classifier(f)

        aligninput = self.stn(featuremap1,featuremap2)
        ax = self.alignbase(aligninput)
        if self.ah:
            ahx = ax
            sx = ahx.size(2) / 4
            kx = ahx.size(2) - sx * 3
            ahx = F.avg_pool2d(ahx, kernel_size=(kx, ahx.size(3)), stride=(sx, ahx.size(3)))
            ahx = self.ahdrop(ahx)
            ahx = self.localah_conv(ahx)
            ahx = self.ahfeat_bn2d(ahx)
            ahx = F.relu(ahx)

            ahx = ahx.chunk(4, 2)
            ahx0 = ahx[0].contiguous().view(ahx[0].size(0), -1)
            ahx1 = ahx[1].contiguous().view(ahx[1].size(0), -1)
            ahx2 = ahx[2].contiguous().view(ahx[2].size(0), -1)
            ahx3 = ahx[3].contiguous().view(ahx[3].size(0), -1)
            ahfeature = torch.cat((ahx0, ahx1, ahx2, ahx3), 1)
            ahnorm = 1. * ahfeature / (torch.norm(ahfeature, 2, dim=-1, keepdim=True).expand_as(ahfeature) + 1e-12)
            ahc0 = self.ahinstance0(ahx0)
            ahc1 = self.ahinstance1(ahx1)
            ahc2 = self.ahinstance2(ahx2)
            ahc3 = self.ahinstance3(ahx3)

        ax = F.avg_pool2d(ax,ax.size()[2:])
        af = ax.view(ax.size(0),-1)
        normaf = 1. * af / (torch.norm(af, 2, dim=-1, keepdim=True).expand_as(af) + 1e-12)
        aout = self.alignclassifier(af)
        if not self.training:
            if not self.gh and not self.ah:
                return torch.cat((normf,normaf),1)
            if self.gh and not self.ah:
                return torch.cat((ghnorm,normaf),1)
            if not self.gh and self.ah:
                return torch.cat((normf,ahnorm),1)
            if self.gh and self.ah:
                return torch.cat((ghnorm,ahnorm),1)

        if not self.gh and not self.ah:
            return (out,aout)
        if self.gh and not self.ah:
            return (ghc0,ghc1,ghc2,ghc3,aout)
        if not self.gh and self.ah:
            return (out,ahc0,ahc1,ahc2,ahc3)
        if self.gh and self.ah:
            return (ghc0,ghc1,ghc2,ghc3,ahc0,ahc1,ahc2,ahc3)
