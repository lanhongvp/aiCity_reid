from __future__ import absolute_import

from .ResNet import *
from .DenseNet import *
from .localDenseNet import *
from .ShuffleNet import *
from .InceptionV4 import *
from .PCB import *

__factory = {
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'densenet121': DenseNet121,
    'shufflenet': ShuffleNet,
    'inceptionv4': InceptionV4ReID,
    'densenet121_local_4':DenseNet121_4,
    'densenet121_local_ap':DenseNet121_ap,
    'densenet121_stn':MyDenseNet_stn,
    'densenet121_stn_local':MyDenseNet_stn_local,
    'PCB':PCBModel,
    'densePCB':DensePCB
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
