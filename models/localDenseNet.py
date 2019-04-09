from __future__ import absolute_import

import torch
from torch.nn import init
from torch import nn
from torch.nn import functional as F
import torchvision
from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d
from IPython import embed

__all__ = ['DenseNet121_4','DenseNet121_ap']

class DenseNet121_4(nn.Module):
    def __init__(self, num_classes, loss = {'softmax'}, num_features=256, dropout = 0.5, hf = False, vf = False, gf = False, **kwargs):
        super(DenseNet121_4, self).__init__()
        self.loss = loss
        self.hf = hf
        self.vf = vf
        self.gf = gf
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.classifier = nn.Linear(1024, num_classes)

        if self.hf:
            # =====================append conv for Horizontal======================= #
            self.localh_conv = nn.Conv2d(1024,self.num_features,kernel_size=1,padding=0,bias=False)
            init.kaiming_normal(self.localh_conv.weight,mode='fan_out')
            self.hfeat_bn2d = nn.BatchNorm2d(self.num_features)
            init.constant(self.hfeat_bn2d.weight,1)
            init.constant(self.hfeat_bn2d.bias,0)

            ##------------------------------stripe----------------------------------------##
            self.hinstance0 = nn.Linear(self.num_features,self.num_classes)
            init.normal(self.hinstance0.weight,std=0.001)
            init.constant(self.hinstance0.bias,0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.hinstance1 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.hinstance1.weight, std=0.001)
            init.constant(self.hinstance1.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.hinstance2 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.hinstance2.weight, std=0.001)
            init.constant(self.hinstance2.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.hinstance3 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.hinstance3.weight, std=0.001)
            init.constant(self.hinstance3.bias, 0)
            ##------------------------------stripe----------------------------------------##

            self.hdrop = nn.Dropout(self.dropout)

        if self.vf:
            # =====================append conv for Horizontal======================= #
            self.localv_conv = nn.Conv2d(1024, self.num_features, kernel_size=1, padding=0, bias=False)
            init.kaiming_normal(self.localv_conv.weight, mode='fan_out')
            self.vfeat_bn2d = nn.BatchNorm2d(self.num_features)
            init.constant(self.vfeat_bn2d.weight, 1)
            init.constant(self.vfeat_bn2d.bias, 0)

            ##------------------------------stripe----------------------------------------##
            self.vinstance0 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.vinstance0.weight, std=0.001)
            init.constant(self.vinstance0.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.vinstance1 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.vinstance1.weight, std=0.001)
            init.constant(self.vinstance1.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.vinstance2 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.vinstance2.weight, std=0.001)
            init.constant(self.vinstance2.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.vinstance3 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.vinstance3.weight, std=0.001)
            init.constant(self.vinstance3.bias, 0)
            ##------------------------------stripe----------------------------------------##

            self.vdrop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.base(x)
        gx = F.avg_pool2d(x,x.size()[2:])
        gf = gx.view(gx.size(0),-1)
        gfnorm = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
        gc = self.classifier(gf)
        hx0 = 0;hx1 = 0;hx2 = 0;hx3 = 0; hnorm = 0
        hc0 = 0;hc1 = 0;hc2 = 0;hc3 = 0
        if self.hf:
            hx = x
            sx = hx.size(2) / 4
            kx = hx.size(2) - sx*3
            hx = F.avg_pool2d(hx,kernel_size=(kx,hx.size(3)),stride=(sx,hx.size(3)))
            # ===================================================================== #

            #out0 = x.view(x.size(0),-1)
            #out0 = x / x.norm(2,1).unsqueeze(1).expand_as(x)
            hx = self.hdrop(hx)
            hx = self.localh_conv(hx)
            hx = self.hfeat_bn2d(hx)
            hx = F.relu(hx)

            hx = hx.chunk(4,2)
            hx0 = hx[0].contiguous().view(hx[0].size(0), -1)
            hx1 = hx[1].contiguous().view(hx[1].size(0), -1)
            hx2 = hx[2].contiguous().view(hx[2].size(0), -1)
            hx3 = hx[3].contiguous().view(hx[3].size(0), -1)
            hfeature = torch.cat((hx0,hx1,hx2,hx3),1)
            hnorm = 1. * hfeature / (torch.norm(hfeature, 2, dim=-1, keepdim=True).expand_as(hfeature) + 1e-12)
            if not self.training:
                if not self.gf and not self.vf:
                    return hnorm
                if self.gf and not self.vf:
                    return torch.cat((gfnorm,hnorm),1)
                if not self.gf and self.vf:
                    pass
                if self.gf and self.vf:
                    pass
            hc0 = self.hinstance0(hx0)
            hc1 = self.hinstance1(hx1)
            hc2 = self.hinstance2(hx2)
            hc3 = self.hinstance3(hx3)

            if not self.gf and not self.vf:
                return (hc0,hc1,hc2,hc3)
            if self.gf and not self.vf:
                return (gc,hc0,hc1,hc2,hc3)
            if not self.gf and self.vf:
                pass
            if self.gf and self.vf:
                pass

        if self.vf:
            vx = x
            sx = vx.size(3) / 4
            kx = vx.size(3) - sx * 3
            vx = F.avg_pool2d(vx, kernel_size=(vx.size(2),kx), stride=(vx.size(2),sx))
            # ===================================================================== #

            # out0 = x.view(x.size(0),-1)
            # out0 = x / x.norm(2,1).unsqueeze(1).expand_as(x)
            vx = self.vdrop(vx)
            vx = self.localv_conv(vx)
            vx = self.vfeat_bn2d(vx)
            vx = F.relu(vx)

            vx = vx.chunk(4, 3)
            vx0 = vx[0].contiguous().view(vx[0].size(0), -1)
            vx1 = vx[1].contiguous().view(vx[1].size(0), -1)
            vx2 = vx[2].contiguous().view(vx[2].size(0), -1)
            vx3 = vx[3].contiguous().view(vx[3].size(0), -1)
            vfeature = torch.cat((vx0,vx1,vx2,vx3),1)
            vnorm = 1. * vfeature / (torch.norm(vfeature, 2, dim=-1, keepdim=True).expand_as(vfeature) + 1e-12)
            if not self.training:
                if not self.gf and not self.hf:
                    return vnorm
                if self.gf and not self.hf:
                    return torch.cat((gfnorm,vnorm), 1)
                if not self.gf and self.hf:
                    return torch.cat((hnorm,vnorm),1)
                if self.gf and self.hf:
                    return torch.cat((gfnorm,hnorm,vnorm), 1)
            vc0 = self.vinstance0(vx0)
            vc1 = self.vinstance1(vx1)
            vc2 = self.vinstance2(vx2)
            vc3 = self.vinstance3(vx3)

            if not self.gf and not self.hf:
                return (vc0,vc1,vc2,vc3)
            if self.gf and not self.hf:
                return (gc, vc0, vc1, vc2, vc3)
            if not self.gf and self.hf:
                return (hc0, hc1, hc2, hc3, vc0, vc1, vc2, vc3)
            if self.gf and self.hf:
                return (gc,hc0,hc1,hc2,hc3,vc0,vc1,vc2,vc3)

class DenseNet121_ap(nn.Module):
    def __init__(self, num_classes, loss = {'softmax'}, num_features=256, dropout = 0.5, hf = False, vf = False, gf = False, **kwargs):
        super(DenseNet121_ap, self).__init__()
        self.loss = loss
        self.hf = hf
        self.vf = vf
        self.gf = gf
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.classifier = nn.Linear(1024, num_classes)

        if self.hf:
            # =====================append conv for Horizontal======================= #
            self.localh_conv = nn.Conv2d(1024,self.num_features,kernel_size=1,padding=0,bias=False)
            init.kaiming_normal(self.localh_conv.weight,mode='fan_out')
            self.hfeat_bn2d = nn.BatchNorm2d(self.num_features)
            init.constant(self.hfeat_bn2d.weight,1)
            init.constant(self.hfeat_bn2d.bias,0)

            ##------------------------------stripe----------------------------------------##
            self.hinstance0 = nn.Linear(self.num_features,self.num_classes)
            init.normal(self.hinstance0.weight,std=0.001)
            init.constant(self.hinstance0.bias,0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.hinstance1 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.hinstance1.weight, std=0.001)
            init.constant(self.hinstance1.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.hinstance2 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.hinstance2.weight, std=0.001)
            init.constant(self.hinstance2.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.hinstance3 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.hinstance3.weight, std=0.001)
            init.constant(self.hinstance3.bias, 0)
            ##------------------------------stripe----------------------------------------##

            self.hdrop = nn.Dropout(self.dropout)

        if self.vf:
            # =====================append conv for Horizontal======================= #
            self.localv_conv = nn.Conv2d(1024, self.num_features, kernel_size=1, padding=0, bias=False)
            init.kaiming_normal(self.localv_conv.weight, mode='fan_out')
            self.vfeat_bn2d = nn.BatchNorm2d(self.num_features)
            init.constant(self.vfeat_bn2d.weight, 1)
            init.constant(self.vfeat_bn2d.bias, 0)

            ##------------------------------stripe----------------------------------------##
            self.vinstance0 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.vinstance0.weight, std=0.001)
            init.constant(self.vinstance0.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.vinstance1 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.vinstance1.weight, std=0.001)
            init.constant(self.vinstance1.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.vinstance2 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.vinstance2.weight, std=0.001)
            init.constant(self.vinstance2.bias, 0)
            ##------------------------------stripe----------------------------------------##
            ##------------------------------stripe----------------------------------------##
            self.vinstance3 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.vinstance3.weight, std=0.001)
            init.constant(self.vinstance3.bias, 0)
            ##------------------------------stripe----------------------------------------##

            self.vdrop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.base(x)
        gx = F.avg_pool2d(x,x.size()[2:])
        gf = gx.view(gx.size(0),-1)
        gfnrom = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
        gc = self.classifier(gf)
        hx0 = 0;hx1 = 0;hx2 = 0;hx3 = 0
        hc0 = 0;hc1 = 0;hc2 = 0;hc3 = 0
        ha0 = 0;ha1 = 0;ha2 = 0;ha3 = 0; hnorm =0
        if self.hf:
            hx = x
            sx = hx.size(2) / 4
            kx = hx.size(2) - sx*3
            hx = F.avg_pool2d(hx,kernel_size=(kx,hx.size(3)),stride=(sx,hx.size(3)))
            # ===================================================================== #
            ha = hx.chunk(4,2)
            ha0 = ha[0].contiguous().view(ha[0].size(0), -1)
            ha1 = ha[1].contiguous().view(ha[1].size(0), -1)
            ha2 = ha[2].contiguous().view(ha[2].size(0), -1)
            ha3 = ha[3].contiguous().view(ha[3].size(0), -1)
            hfeature = torch.cat((ha0,ha1,ha2,ha3),1)
            hnorm = 1. * hfeature / (torch.norm(hfeature, 2, dim=-1, keepdim=True).expand_as(hfeature) + 1e-12)
            #out0 = x.view(x.size(0),-1)
            #out0 = x / x.norm(2,1).unsqueeze(1).expand_as(x)
            hx = self.hdrop(hx)
            hx = self.localh_conv(hx)
            hx = self.hfeat_bn2d(hx)
            hx = F.relu(hx)

            hx = hx.chunk(4,2)
            hx0 = hx[0].contiguous().view(hx[0].size(0), -1)
            hx1 = hx[1].contiguous().view(hx[1].size(0), -1)
            hx2 = hx[2].contiguous().view(hx[2].size(0), -1)
            hx3 = hx[3].contiguous().view(hx[3].size(0), -1)
            if not self.training:
                if not self.gf and not self.vf:
                    return hnorm
                if self.gf and not self.vf:
                    return torch.cat((gfnrom,hnorm),1)
                if not self.gf and self.vf:
                    pass
                if self.gf and self.vf:
                    pass
            hc0 = self.hinstance0(hx0)
            hc1 = self.hinstance1(hx1)
            hc2 = self.hinstance2(hx2)
            hc3 = self.hinstance3(hx3)

            if not self.gf and not self.vf:
                return (hc0,hc1,hc2,hc3)
            if self.gf and not self.vf:
                return (gc,hc0,hc1,hc2,hc3)
            if not self.gf and self.vf:
                pass
            if self.gf and self.vf:
                pass

        if self.vf:
            vx = x
            sx = vx.size(3) / 4
            kx = vx.size(3) - sx * 3
            vx = F.avg_pool2d(vx, kernel_size=(vx.size(2),kx), stride=(vx.size(2),sx))
            # ===================================================================== #
            va = vx.chunk(4,3)
            va0 = va[0].contiguous().view(va[0].size(0), -1)
            va1 = va[1].contiguous().view(va[1].size(0), -1)
            va2 = va[2].contiguous().view(va[2].size(0), -1)
            va3 = va[3].contiguous().view(va[3].size(0), -1)
            vfeature = torch.cat((va0,va1,va2,va3),1)
            vnorm = 1. * vfeature / (torch.norm(vfeature, 2, dim=-1, keepdim=True).expand_as(vfeature) + 1e-12)
            # out0 = x.view(x.size(0),-1)
            # out0 = x / x.norm(2,1).unsqueeze(1).expand_as(x)
            vx = self.vdrop(vx)
            vx = self.localv_conv(vx)
            vx = self.vfeat_bn2d(vx)
            vx = F.relu(vx)

            vx = vx.chunk(4, 3)
            vx0 = vx[0].contiguous().view(vx[0].size(0), -1)
            vx1 = vx[1].contiguous().view(vx[1].size(0), -1)
            vx2 = vx[2].contiguous().view(vx[2].size(0), -1)
            vx3 = vx[3].contiguous().view(vx[3].size(0), -1)
            if not self.training:
                if not self.gf and not self.hf:
                    return vnorm
                if self.gf and not self.hf:
                    return torch.cat((gfnrom,vnorm), 1)
                if not self.gf and self.hf:
                    return torch.cat((hnorm,vnorm),1)
                if self.gf and self.hf:
                    return torch.cat((gfnrom,hnorm,vnorm), 1)
            vc0 = self.vinstance0(vx0)
            vc1 = self.vinstance1(vx1)
            vc2 = self.vinstance2(vx2)
            vc3 = self.vinstance3(vx3)

            if not self.gf and not self.hf:
                return (vc0,vc1,vc2,vc3)
            if self.gf and not self.hf:
                return (gc, vc0, vc1, vc2, vc3)
            if not self.gf and self.hf:
                return (hc0, hc1, hc2, hc3, vc0, vc1, vc2, vc3)
            if self.gf and self.hf:
                return (gc,hc0,hc1,hc2,hc3,vc0,vc1,vc2,vc3)