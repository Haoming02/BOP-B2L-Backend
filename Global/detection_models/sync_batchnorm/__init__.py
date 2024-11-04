# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

from .batchnorm import set_sbn_eps_mode
from .batchnorm import (
    SynchronizedBatchNorm1d,
    SynchronizedBatchNorm2d,
    SynchronizedBatchNorm3d,
)
from .batchnorm import patch_sync_batchnorm, convert_model
from .replicate import DataParallelWithCallback, patch_replication_callback
