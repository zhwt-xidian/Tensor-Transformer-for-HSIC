<<<<<<< HEAD
'''
    LibTNN
    本库包含以下层与函数：
    2. Tucker_TRL(Tucker Tensor Regression Layer)   Tucker分解张量回归层
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import tensorly as tl
from tensorly.tenalg import inner
from tensorly.decomposition import partial_tucker
from tensorly.decomposition import parafac

tl.set_backend('pytorch')

class Tucker_TRL(nn.Module):
    
    '''
    TRL(Tucker Tensor Regression Layer) 层的实现
    实现思想:
    对于一个大小为 (batch_size, m1, m2, m3) 大小的激活张量，网络回归希望得到一个大小为 (batch_size, class_num) 的输出
    为了得到和预期输出，可以使用一个大小为 (m1, m2, m3, class_num) 的回归权重张量，采用广义内积 (Generalized inner-product) 的形式，得到输出
    回归权重张量的大小过大，不便于计算，可以使用 Tucker 分解的思想，使用秩 (a, b, c, d) 的 Tucker 分解
    核心张量大小为 (a, b, c, d)，分解矩阵大小分别为 (m1, a) (m2, b) (m3, c) (class_num, d)，简化计算
    '''
    
    def __init__(self, input_size: tuple, output_size: tuple, ranks: tuple, verbose=1, **kwargs):
        
        '''
            input_size: tuple   为输入激活张量的大小 (batch_size, m1, m2, m3)，其第一个参数 batch_size 无效
            output_size: tuple  为输出矩阵的大小 (batch_size, class_num)，其第一个参数 batch_size 无效
            ranks: tuple        为回归张量 Tucker 分解核心张量大小
        '''
        
        super(Tucker_TRL, self).__init__(**kwargs)
        
        self.ranks = list(ranks)
        self.verbose = verbose

        # 参数列表化
        self.input_size = list(input_size)
        self.output_size = list(output_size)
        
        # 输出类别数
        self.n_outputs = int(np.prod(output_size[1:]))

        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)      # 初始化 Tucker 回归核心张量（权重）
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)               # 偏置
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])    # 分解矩阵 dim=0 方向大小 (m1, m2, m3, class_num)

        self.factors = []   # 初始化分解矩阵队列
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):       # zip 包装成元组，并使用枚举
            # (m1, a) (m2, b) (m3, c) (class_num, d)
            self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.register_parameter('factor_{}'.format(index), self.factors[index])
        
        self.core.data.uniform_(-0.1, 0.1)      # 重新分布核心张量
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)          # 重新分布分解张量

    def forward(self, x):           # 前向传播
        regression_weights = tl.tucker_to_tensor((self.core, self.factors))
        return inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias
=======
'''
    LibTNN
    本库包含以下层与函数：
    2. Tucker_TRL(Tucker Tensor Regression Layer)   Tucker分解张量回归层
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import tensorly as tl
from tensorly.tenalg import inner
from tensorly.decomposition import partial_tucker
from tensorly.decomposition import parafac

tl.set_backend('pytorch')

class Tucker_TRL(nn.Module):
    
    '''
    TRL(Tucker Tensor Regression Layer) 层的实现
    实现思想:
    对于一个大小为 (batch_size, m1, m2, m3) 大小的激活张量，网络回归希望得到一个大小为 (batch_size, class_num) 的输出
    为了得到和预期输出，可以使用一个大小为 (m1, m2, m3, class_num) 的回归权重张量，采用广义内积 (Generalized inner-product) 的形式，得到输出
    回归权重张量的大小过大，不便于计算，可以使用 Tucker 分解的思想，使用秩 (a, b, c, d) 的 Tucker 分解
    核心张量大小为 (a, b, c, d)，分解矩阵大小分别为 (m1, a) (m2, b) (m3, c) (class_num, d)，简化计算
    '''
    
    def __init__(self, input_size: tuple, output_size: tuple, ranks: tuple, verbose=1, **kwargs):
        
        '''
            input_size: tuple   为输入激活张量的大小 (batch_size, m1, m2, m3)，其第一个参数 batch_size 无效
            output_size: tuple  为输出矩阵的大小 (batch_size, class_num)，其第一个参数 batch_size 无效
            ranks: tuple        为回归张量 Tucker 分解核心张量大小
        '''
        
        super(Tucker_TRL, self).__init__(**kwargs)
        
        self.ranks = list(ranks)
        self.verbose = verbose

        # 参数列表化
        self.input_size = list(input_size)
        self.output_size = list(output_size)
        
        # 输出类别数
        self.n_outputs = int(np.prod(output_size[1:]))

        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)      # 初始化 Tucker 回归核心张量（权重）
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)               # 偏置
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])    # 分解矩阵 dim=0 方向大小 (m1, m2, m3, class_num)

        self.factors = []   # 初始化分解矩阵队列
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):       # zip 包装成元组，并使用枚举
            # (m1, a) (m2, b) (m3, c) (class_num, d)
            self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.register_parameter('factor_{}'.format(index), self.factors[index])
        
        self.core.data.uniform_(-0.1, 0.1)      # 重新分布核心张量
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)          # 重新分布分解张量

    def forward(self, x):           # 前向传播
        regression_weights = tl.tucker_to_tensor((self.core, self.factors))
        return inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias
>>>>>>> 28d2ef547ea47d584fd9708984d528f099f4b9da
