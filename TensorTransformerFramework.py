<<<<<<< HEAD
import torch
import torch.nn as nn
import importlib.util
pyc_path = '__pycache__/utils.cpython-38.pyc'  
spec = importlib.util.spec_from_file_location("utils", pyc_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
import tensorly as tl
import numpy as np
from LibTNN import Tucker_TRL
from itertools import chain


tl.set_backend('pytorch')

def transformer_normalize(in_tensor):
    tensor_mean = in_tensor.mean(dim=(1, 2, 3), keepdim=True)
    tensor_std = in_tensor.std(dim=(1, 2, 3), keepdim=True)

    out_tensor = (in_tensor - tensor_mean) / tensor_std

    return out_tensor

# 经典的Self-Attention
class selfAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads):
        super(selfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)  # Q
        self.key = nn.Linear(hidden_size, self.all_head_size)  # K
        self.value = nn.Linear(hidden_size, self.all_head_size)  # V

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # 使用全连接层模拟 QKV 变换
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose(mixed_query_layer)  # (8,267,4)-->(8,1,267,4)
        key_layer = self.transpose(mixed_key_layer)
        value_layer = self.transpose(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)  # 8 1 267 4
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 8 267 1 4
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # 8 267 4
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

# 不带MLP的3D Self-Attention
class Self3DAttention(nn.Module):

    def __init__(self, win_size, dims, channels):
        super(Self3DAttention, self).__init__()

        self.win_size = win_size
        self.dims = dims
        self.channels = channels

        self.query = []
        self.key = []
        self.value = []

        # 开始遍历，并注册q k v矩阵的参数
        self.query.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(0), self.query[0])  # 注册参数
        self.query.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(1), self.query[1])  # 注册参数
        self.query.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(2), self.query[2])  # 注册参数

        self.key.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(0), self.key[0])  # 注册参数
        self.key.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(1), self.key[1])  # 注册参数
        self.key.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(2), self.key[2])  # 注册参数

        self.value.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(0), self.value[0])  # 注册参数
        self.value.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(1), self.value[1])  # 注册参数
        self.value.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(2), self.value[2])  # 注册参数

        for q in self.query:
            q.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for k in self.key:
            k.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for v in self.value:
            v.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # tucker分解原生就是支持batch的
    def q_tucker(self, x):
        for index, query in enumerate(self.query):
            x = tl.tenalg.mode_dot(x, query, mode=index + 1)
        return x

    def k_tucker(self, x):
        for index, key in enumerate(self.key):
            x = tl.tenalg.mode_dot(x, key, mode=index + 1)
        return x

    def v_tucker(self, x):
        for index, value in enumerate(self.value):
            x = tl.tenalg.mode_dot(x, value, mode=index + 1)
        return x

    def forward(self, x):
        q = x.clone()
        k = x.clone()
        v = x.clone()
        for i in range(self.channels):
            q[:, :, :, :, i] = self.q_tucker(q[:, :, :, :, i])
            k[:, :, :, :, i] = self.k_tucker(k[:, :, :, :, i])
            v[:, :, :, :, i] = self.v_tucker(v[:, :, :, :, i])
        q = torch.reshape(q, (-1, self.dims * self.win_size * self.win_size, self.channels))
        k = torch.reshape(k, (-1, self.dims * self.win_size * self.win_size, self.channels))
        attention_scores = torch.matmul(q.transpose(-1, -2), k)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        v = torch.reshape(v, (-1, self.channels, self.dims * self.win_size * self.win_size))
        context_layer = torch.matmul(attention_probs, v)
        context_layer = torch.reshape(context_layer, x.size())
        
        return context_layer
        

# 2024.7.25
# 简化版单层Encoder
class Self3DAttentionToken_r(nn.Module):
    def __init__(self, win_size, dims, channels):
        super(Self3DAttentionToken_r, self).__init__()
        self.win_size = win_size
        self.dims = dims
        self.channels = channels

        # 初始化并注册query, key, value和MLP的参数
        self.query = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                      [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.key = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                    [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.value = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                      [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.MLP = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                    [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])

        # 初始化参数
        for param in chain(self.query, self.key, self.value, self.MLP):
            param.data.uniform_(-0.1, 0.1)

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def tucker_decomposition(self, x, parameters):
        for index, param in enumerate(parameters):
            x = tl.tenalg.mode_dot(x, param, mode=index + 1)
        return x

    def forward(self, x):
        x = transformer_normalize(x)
        q, k, v = x.clone(), x.clone(), x.clone()

        for i in range(self.channels + 1):
            q[..., i] = self.tucker_decomposition(q[..., i], self.query)
            k[..., i] = self.tucker_decomposition(k[..., i], self.key)
            v[..., i] = self.tucker_decomposition(v[..., i], self.value)

        q = q.view(-1, self.dims * self.win_size * self.win_size, self.channels + 1)
        k = k.view(-1, self.dims * self.win_size * self.win_size, self.channels + 1)
        attention_scores = torch.matmul(q.transpose(-1, -2), k)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        v = v.view(-1, self.channels + 1, self.dims * self.win_size * self.win_size)
        context_layer = torch.matmul(attention_probs, v).view(x.size())

        x = x + context_layer
        x = transformer_normalize(x)

        for i in range(self.channels + 1):
            x[..., i] = self.tucker_decomposition(x[..., i], self.MLP)
        return x + context_layer
    

# 2024.7.29
# 多头Encoder
class Self3DAttentionToken_rh(nn.Module):
    def __init__(self, win_size, dims, channels, heads):
        super(Self3DAttentionToken_rh, self).__init__()
        self.win_size = win_size
        self.dims = dims
        self.channels = channels
        self.heads = heads

        # 确保头的数量和维度的除法结果为整数
        assert dims % heads == 0, "dims must be divisible by heads"

        # 初始化并注册query, key, value和MLP的参数
        self.query = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                      [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.key = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                    [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.value = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                      [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.MLP = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                    [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])

        self.attn_drop = nn.Dropout()

        # 初始化参数
        for param in chain(self.query, self.key, self.value, self.MLP):
            param.data.uniform_(-0.1, 0.1)

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def tucker_decomposition(self, x, parameters):
        for index, param in enumerate(parameters):
            x = tl.tenalg.mode_dot(x, param, mode=index + 1)
        return x

    def forward(self, x):
        x = transformer_normalize(x)
        q, k, v = x.clone(), x.clone(), x.clone()

        for i in range(self.channels + 1):
            q[..., i] = self.tucker_decomposition(q[..., i], self.query)
            k[..., i] = self.tucker_decomposition(k[..., i], self.key)
            v[..., i] = self.tucker_decomposition(v[..., i], self.value)
        
        # todo 改写成 8,7,7,54,5 -> 8,7,7,1,54,5 -> (8,1,5,7,7,54) -> (8,1,5,7*7*54)
        q = q.view(-1, self.win_size, self.win_size, self.heads, self.dims // self.heads, self.channels + 1).permute(0,3,5,1,2,4).reshape(-1,self.heads,self.channels+1,self.win_size*self.win_size*(self.dims // self.heads))
        k = k.view(-1, self.win_size, self.win_size, self.heads, self.dims // self.heads, self.channels + 1).permute(0,3,5,1,2,4).reshape(-1,self.heads,self.channels+1,self.win_size*self.win_size*(self.dims // self.heads))
        attn = (q @ k.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        v = v.view(-1, self.win_size, self.win_size, self.heads, self.dims // self.heads, self.channels + 1).permute(0,3,5,1,2,4).reshape(-1,self.heads,self.channels+1,self.win_size*self.win_size*(self.dims // self.heads))
        context_layer = (attn @ v).reshape(-1,self.heads,self.channels+1,self.win_size,self.win_size,self.dims // self.heads).permute(0,3,4,1,5,2).reshape(-1,self.win_size,self.win_size,self.dims,self.channels+1)

        x = x + context_layer
        x = transformer_normalize(x)

        for i in range(self.channels + 1):
            x[..., i] = self.tucker_decomposition(x[..., i], self.MLP)
        return x + context_layer
    
    
# 带MLP的3D Self-Attention
class Self3DAttentionToken(nn.Module):

    def __init__(self, win_size, dims, channels):
        super(Self3DAttentionToken, self).__init__()
        self.win_size = win_size
        self.dims = dims
        self.channels = channels

        self.query = []
        self.key = []
        self.value = []
        self.MLP = []

        # 开始遍历，并注册q k v矩阵的参数
        # 注册q矩阵的参数
        self.query.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(0), self.query[0])  # 注册参数
        self.query.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(1), self.query[1])  # 注册参数
        self.query.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(2), self.query[2])  # 注册参数
        # 注册k矩阵的参数
        self.key.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(0), self.key[0])  # 注册参数
        self.key.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(1), self.key[1])  # 注册参数
        self.key.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(2), self.key[2])  # 注册参数
        # 注册v矩阵的参数
        self.value.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(0), self.value[0])  # 注册参数
        self.value.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(1), self.value[1])  # 注册参数
        self.value.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(2), self.value[2])  # 注册参数
        # 注册MLP矩阵的参数，这其实是Encoder层的MLP层
        self.MLP.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('mlp_{}'.format(0), self.MLP[0])  # 注册参数
        self.MLP.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('mlp_{}'.format(1), self.MLP[1])  # 注册参数
        self.MLP.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('mlp_{}'.format(2), self.MLP[2])  # 注册参数

        for q in self.query:
            q.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for k in self.key:
            k.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for v in self.value:
            v.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for m in self.MLP:
            m.data.uniform_(-0.1, 0.1)

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # tucker分解原生就是支持batch的
    def q_tucker(self, x):
        for index, query in enumerate(self.query):
            x = tl.tenalg.mode_dot(x, query, mode=index + 1)
        return x

    def k_tucker(self, x):
        for index, key in enumerate(self.key):
            x = tl.tenalg.mode_dot(x, key, mode=index + 1)
        return x

    def v_tucker(self, x):
        for index, value in enumerate(self.value):
            x = tl.tenalg.mode_dot(x, value, mode=index + 1)
        return x

    def m_tucker(self, x):
        for index, mlp in enumerate(self.MLP):
            x = tl.tenalg.mode_dot(x, mlp, mode=index + 1)
        return x

    def forward(self, x):
        # todo 应该先对x归一化一下    ----done
        x = transformer_normalize(x)
        q = x.clone()
        k = x.clone()
        v = x.clone()
        # x---> (8,5,5,36,7)
        for i in range(self.channels + 1):
            q[:, :, :, :, i] = self.q_tucker(q[:, :, :, :, i])
            k[:, :, :, :, i] = self.k_tucker(k[:, :, :, :, i])
            v[:, :, :, :, i] = self.v_tucker(v[:, :, :, :, i])
        # q*k--->(8,1,7,7)
        q = torch.reshape(q, (-1, self.dims * self.win_size * self.win_size, self.channels + 1))
        k = torch.reshape(k, (-1, self.dims * self.win_size * self.win_size, self.channels + 1))
        attention_scores = torch.matmul(q.transpose(-1, -2), k)
        # (8,7,7)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        v = torch.reshape(v, (-1, self.channels + 1, self.dims * self.win_size * self.win_size))
        context_layer = torch.matmul(attention_probs, v)
        context_layer = torch.reshape(context_layer, x.size())

        # 在这里做完单层encoder的操作得了
        x = x + context_layer
        x = transformer_normalize(x)
        temp = x.clone()
        # 使用张量变换来代替MLP层
        for i in range(self.channels + 1):
            x[:, :, :, :, i] = self.m_tucker(x[:, :, :, :, i])
        return temp + x

# Encoder层
class TensorEncoder(nn.Module):
    def __init__(self, winsize, dim, channels, layer):
        # dim是单个样本的第三维度大小，channels是有多少个样本，layer是Encoder层数
        super(TensorEncoder, self).__init__()
        # 根据层数，来创建TensorAttention的个数
        self.encoders = nn.Sequential(*[
            Self3DAttentionToken(winsize, dim, channels)
            # Self3DAttentionToken_r(winsize, dim, channels)
            # Self3DAttentionToken_rh(winsize, dim, channels, 3)
            for i in range(layer)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, winsize, winsize, dim, 1))  # (1,15,15,36,1)
        self.pos_embed = nn.Parameter(torch.zeros(1, winsize, winsize, dim, channels + 1))  # (1,15,15,36,8)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1, -1, -1)
        x = torch.cat((cls_token, x), dim=-1)
        x = x + self.pos_embed
        x = self.encoders(x)
        return x[:, :, :, :, 1]
        # return x[:, :, :, :, 0]
    
# LK，spectral length：54，stride：0
class TensorTransformerHSI_unoverlapping_LK(nn.Module):
    def __init__(self):
        super(TensorTransformerHSI_unoverlapping_LK, self).__init__()
        self.encoder = TensorEncoder(7, 54, 5, 12)
        self.classifier = nn.Sequential(
            nn.Linear(54 * 7 * 7, 9),
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, 7, 7, 54, 5))
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
# LK，spectral length：54，stride：27
class TensorTransformerHSI_overlapping_LK(nn.Module):
    def __init__(self):
        super(TensorTransformerHSI_overlapping_LK, self).__init__()
        self.win = 7
        self.embed_dim = 54
        self.seq_num = 9
        self.encoder = TensorEncoder(self.win, self.embed_dim, self.seq_num, 12)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * self.win * self.win, 9),
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.win, self.win, 1, 270)) # (7,7,270) --> (7,7,1,270)
        step = int((x.shape[4] - self.embed_dim) / (self.seq_num - 1))  # stride = 27

        x_samples = torch.zeros(x.shape[0], self.win, self.win, self.seq_num, self.embed_dim).to('cuda:0')
        for i in range(self.seq_num):
            start_index = i * step
            end_index = start_index + self.embed_dim
            if end_index <= x.shape[-1]:  # 确保索引不会超出范围
                x_samples[..., i, :] = x[..., 0, start_index:end_index]
        x_samples = x_samples.permute(0, 1, 2, 4, 3)

        x = self.encoder(x_samples)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
class TensorTransformerHSI_unoverlapping_Hou(nn.Module):
    def __init__(self):
        super(TensorTransformerHSI_unoverlapping_Hou, self).__init__()
        self.win = 7
        self.encoder = TensorEncoder(self.win, 36, 4, 12)
        self.classifier = nn.Sequential(
            nn.Linear(36 * self.win * self.win, 15),
        )

    def forward(self, x):
        # (7,7,144) --> (7,7,36,4)
        x = torch.reshape(x, (-1, self.win, self.win, 36, 4))
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
# Hou： spectral length：54，stride：30
class TensorTransformerHSI_overlapping_Hou(nn.Module):
    def __init__(self):
        super(TensorTransformerHSI_overlapping_Hou, self).__init__()
        self.win = 7
        self.embed_dim = 54
        self.seq_num = 4
        self.encoder = TensorEncoder(self.win, self.embed_dim, self.seq_num, 12)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * self.win * self.win, 15),
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.win, self.win, 1, 144)) #(9,9,144) --> (9,9,1,144)
        step = int((x.shape[4] - self.embed_dim) / (self.seq_num - 1))

        x_samples = torch.zeros(x.shape[0], self.win, self.win, self.seq_num, self.embed_dim)#.to('cuda:0')
        for i in range(self.seq_num):
            start_index = i * step
            end_index = start_index + self.embed_dim
            if end_index <= x.shape[-1]:  # 确保索引不会超出范围
                x_samples[..., i, :] = x[..., 0, start_index:end_index]
        x_samples = x_samples.permute(0, 1, 2, 4, 3)

        x = self.encoder(x_samples)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    x = torch.randn(size=(8, 7, 7, 270))
    net = TensorTransformerHSI_unoverlapping_LK()
    # net = TensorTransformerHSI_overlapping_LK()

    # x = torch.randn(size=(8, 7, 7, 144))
    # net = TensorTransformerHSI_unoverlapping_Hou()
    # net = TensorTransformerHSI_overlapping_Hou()

    output = net(x)
=======
import torch
import torch.nn as nn
import importlib.util
pyc_path = '__pycache__/utils.cpython-38.pyc'  
spec = importlib.util.spec_from_file_location("utils", pyc_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
import tensorly as tl
import numpy as np
from LibTNN import Tucker_TRL
from itertools import chain


tl.set_backend('pytorch')

def transformer_normalize(in_tensor):
    tensor_mean = in_tensor.mean(dim=(1, 2, 3), keepdim=True)
    tensor_std = in_tensor.std(dim=(1, 2, 3), keepdim=True)

    out_tensor = (in_tensor - tensor_mean) / tensor_std

    return out_tensor

# 经典的Self-Attention
class selfAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads):
        super(selfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)  # Q
        self.key = nn.Linear(hidden_size, self.all_head_size)  # K
        self.value = nn.Linear(hidden_size, self.all_head_size)  # V

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # 使用全连接层模拟 QKV 变换
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose(mixed_query_layer)  # (8,267,4)-->(8,1,267,4)
        key_layer = self.transpose(mixed_key_layer)
        value_layer = self.transpose(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)  # 8 1 267 4
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 8 267 1 4
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # 8 267 4
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

# 不带MLP的3D Self-Attention
class Self3DAttention(nn.Module):

    def __init__(self, win_size, dims, channels):
        super(Self3DAttention, self).__init__()

        self.win_size = win_size
        self.dims = dims
        self.channels = channels

        self.query = []
        self.key = []
        self.value = []

        # 开始遍历，并注册q k v矩阵的参数
        self.query.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(0), self.query[0])  # 注册参数
        self.query.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(1), self.query[1])  # 注册参数
        self.query.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(2), self.query[2])  # 注册参数

        self.key.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(0), self.key[0])  # 注册参数
        self.key.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(1), self.key[1])  # 注册参数
        self.key.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(2), self.key[2])  # 注册参数

        self.value.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(0), self.value[0])  # 注册参数
        self.value.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(1), self.value[1])  # 注册参数
        self.value.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(2), self.value[2])  # 注册参数

        for q in self.query:
            q.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for k in self.key:
            k.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for v in self.value:
            v.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # tucker分解原生就是支持batch的
    def q_tucker(self, x):
        for index, query in enumerate(self.query):
            x = tl.tenalg.mode_dot(x, query, mode=index + 1)
        return x

    def k_tucker(self, x):
        for index, key in enumerate(self.key):
            x = tl.tenalg.mode_dot(x, key, mode=index + 1)
        return x

    def v_tucker(self, x):
        for index, value in enumerate(self.value):
            x = tl.tenalg.mode_dot(x, value, mode=index + 1)
        return x

    def forward(self, x):
        q = x.clone()
        k = x.clone()
        v = x.clone()
        for i in range(self.channels):
            q[:, :, :, :, i] = self.q_tucker(q[:, :, :, :, i])
            k[:, :, :, :, i] = self.k_tucker(k[:, :, :, :, i])
            v[:, :, :, :, i] = self.v_tucker(v[:, :, :, :, i])
        q = torch.reshape(q, (-1, self.dims * self.win_size * self.win_size, self.channels))
        k = torch.reshape(k, (-1, self.dims * self.win_size * self.win_size, self.channels))
        attention_scores = torch.matmul(q.transpose(-1, -2), k)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        v = torch.reshape(v, (-1, self.channels, self.dims * self.win_size * self.win_size))
        context_layer = torch.matmul(attention_probs, v)
        context_layer = torch.reshape(context_layer, x.size())
        
        return context_layer
        

# 2024.7.25
# 简化版单层Encoder
class Self3DAttentionToken_r(nn.Module):
    def __init__(self, win_size, dims, channels):
        super(Self3DAttentionToken_r, self).__init__()
        self.win_size = win_size
        self.dims = dims
        self.channels = channels

        # 初始化并注册query, key, value和MLP的参数
        self.query = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                      [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.key = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                    [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.value = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                      [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.MLP = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                    [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])

        # 初始化参数
        for param in chain(self.query, self.key, self.value, self.MLP):
            param.data.uniform_(-0.1, 0.1)

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def tucker_decomposition(self, x, parameters):
        for index, param in enumerate(parameters):
            x = tl.tenalg.mode_dot(x, param, mode=index + 1)
        return x

    def forward(self, x):
        x = transformer_normalize(x)
        q, k, v = x.clone(), x.clone(), x.clone()

        for i in range(self.channels + 1):
            q[..., i] = self.tucker_decomposition(q[..., i], self.query)
            k[..., i] = self.tucker_decomposition(k[..., i], self.key)
            v[..., i] = self.tucker_decomposition(v[..., i], self.value)

        q = q.view(-1, self.dims * self.win_size * self.win_size, self.channels + 1)
        k = k.view(-1, self.dims * self.win_size * self.win_size, self.channels + 1)
        attention_scores = torch.matmul(q.transpose(-1, -2), k)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        v = v.view(-1, self.channels + 1, self.dims * self.win_size * self.win_size)
        context_layer = torch.matmul(attention_probs, v).view(x.size())

        x = x + context_layer
        x = transformer_normalize(x)

        for i in range(self.channels + 1):
            x[..., i] = self.tucker_decomposition(x[..., i], self.MLP)
        return x + context_layer
    

# 2024.7.29
# 多头Encoder
class Self3DAttentionToken_rh(nn.Module):
    def __init__(self, win_size, dims, channels, heads):
        super(Self3DAttentionToken_rh, self).__init__()
        self.win_size = win_size
        self.dims = dims
        self.channels = channels
        self.heads = heads

        # 确保头的数量和维度的除法结果为整数
        assert dims % heads == 0, "dims must be divisible by heads"

        # 初始化并注册query, key, value和MLP的参数
        self.query = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                      [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.key = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                    [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.value = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                      [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])
        self.MLP = nn.ParameterList([nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True) for _ in range(2)] +
                                    [nn.Parameter(tl.zeros((dims, dims)), requires_grad=True)])

        self.attn_drop = nn.Dropout()

        # 初始化参数
        for param in chain(self.query, self.key, self.value, self.MLP):
            param.data.uniform_(-0.1, 0.1)

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def tucker_decomposition(self, x, parameters):
        for index, param in enumerate(parameters):
            x = tl.tenalg.mode_dot(x, param, mode=index + 1)
        return x

    def forward(self, x):
        x = transformer_normalize(x)
        q, k, v = x.clone(), x.clone(), x.clone()

        for i in range(self.channels + 1):
            q[..., i] = self.tucker_decomposition(q[..., i], self.query)
            k[..., i] = self.tucker_decomposition(k[..., i], self.key)
            v[..., i] = self.tucker_decomposition(v[..., i], self.value)
        
        # todo 改写成 8,7,7,54,5 -> 8,7,7,1,54,5 -> (8,1,5,7,7,54) -> (8,1,5,7*7*54)
        q = q.view(-1, self.win_size, self.win_size, self.heads, self.dims // self.heads, self.channels + 1).permute(0,3,5,1,2,4).reshape(-1,self.heads,self.channels+1,self.win_size*self.win_size*(self.dims // self.heads))
        k = k.view(-1, self.win_size, self.win_size, self.heads, self.dims // self.heads, self.channels + 1).permute(0,3,5,1,2,4).reshape(-1,self.heads,self.channels+1,self.win_size*self.win_size*(self.dims // self.heads))
        attn = (q @ k.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        v = v.view(-1, self.win_size, self.win_size, self.heads, self.dims // self.heads, self.channels + 1).permute(0,3,5,1,2,4).reshape(-1,self.heads,self.channels+1,self.win_size*self.win_size*(self.dims // self.heads))
        context_layer = (attn @ v).reshape(-1,self.heads,self.channels+1,self.win_size,self.win_size,self.dims // self.heads).permute(0,3,4,1,5,2).reshape(-1,self.win_size,self.win_size,self.dims,self.channels+1)

        x = x + context_layer
        x = transformer_normalize(x)

        for i in range(self.channels + 1):
            x[..., i] = self.tucker_decomposition(x[..., i], self.MLP)
        return x + context_layer
    
    
# 带MLP的3D Self-Attention
class Self3DAttentionToken(nn.Module):

    def __init__(self, win_size, dims, channels):
        super(Self3DAttentionToken, self).__init__()
        self.win_size = win_size
        self.dims = dims
        self.channels = channels

        self.query = []
        self.key = []
        self.value = []
        self.MLP = []

        # 开始遍历，并注册q k v矩阵的参数
        # 注册q矩阵的参数
        self.query.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(0), self.query[0])  # 注册参数
        self.query.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(1), self.query[1])  # 注册参数
        self.query.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('query_{}'.format(2), self.query[2])  # 注册参数
        # 注册k矩阵的参数
        self.key.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(0), self.key[0])  # 注册参数
        self.key.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(1), self.key[1])  # 注册参数
        self.key.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('key_{}'.format(2), self.key[2])  # 注册参数
        # 注册v矩阵的参数
        self.value.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(0), self.value[0])  # 注册参数
        self.value.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(1), self.value[1])  # 注册参数
        self.value.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('value_{}'.format(2), self.value[2])  # 注册参数
        # 注册MLP矩阵的参数，这其实是Encoder层的MLP层
        self.MLP.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('mlp_{}'.format(0), self.MLP[0])  # 注册参数
        self.MLP.append(nn.Parameter(tl.zeros((win_size, win_size)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('mlp_{}'.format(1), self.MLP[1])  # 注册参数
        self.MLP.append(nn.Parameter(tl.zeros((dims, dims)), requires_grad=True))  # 生成权重矩阵
        self.register_parameter('mlp_{}'.format(2), self.MLP[2])  # 注册参数

        for q in self.query:
            q.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for k in self.key:
            k.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for v in self.value:
            v.data.uniform_(-0.1, 0.1)  # 重新分布分解张量

        for m in self.MLP:
            m.data.uniform_(-0.1, 0.1)

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # tucker分解原生就是支持batch的
    def q_tucker(self, x):
        for index, query in enumerate(self.query):
            x = tl.tenalg.mode_dot(x, query, mode=index + 1)
        return x

    def k_tucker(self, x):
        for index, key in enumerate(self.key):
            x = tl.tenalg.mode_dot(x, key, mode=index + 1)
        return x

    def v_tucker(self, x):
        for index, value in enumerate(self.value):
            x = tl.tenalg.mode_dot(x, value, mode=index + 1)
        return x

    def m_tucker(self, x):
        for index, mlp in enumerate(self.MLP):
            x = tl.tenalg.mode_dot(x, mlp, mode=index + 1)
        return x

    def forward(self, x):
        # todo 应该先对x归一化一下    ----done
        x = transformer_normalize(x)
        q = x.clone()
        k = x.clone()
        v = x.clone()
        # x---> (8,5,5,36,7)
        for i in range(self.channels + 1):
            q[:, :, :, :, i] = self.q_tucker(q[:, :, :, :, i])
            k[:, :, :, :, i] = self.k_tucker(k[:, :, :, :, i])
            v[:, :, :, :, i] = self.v_tucker(v[:, :, :, :, i])
        # q*k--->(8,1,7,7)
        q = torch.reshape(q, (-1, self.dims * self.win_size * self.win_size, self.channels + 1))
        k = torch.reshape(k, (-1, self.dims * self.win_size * self.win_size, self.channels + 1))
        attention_scores = torch.matmul(q.transpose(-1, -2), k)
        # (8,7,7)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        v = torch.reshape(v, (-1, self.channels + 1, self.dims * self.win_size * self.win_size))
        context_layer = torch.matmul(attention_probs, v)
        context_layer = torch.reshape(context_layer, x.size())

        # 在这里做完单层encoder的操作得了
        x = x + context_layer
        x = transformer_normalize(x)
        temp = x.clone()
        # 使用张量变换来代替MLP层
        for i in range(self.channels + 1):
            x[:, :, :, :, i] = self.m_tucker(x[:, :, :, :, i])
        return temp + x

# Encoder层
class TensorEncoder(nn.Module):
    def __init__(self, winsize, dim, channels, layer):
        # dim是单个样本的第三维度大小，channels是有多少个样本，layer是Encoder层数
        super(TensorEncoder, self).__init__()
        # 根据层数，来创建TensorAttention的个数
        self.encoders = nn.Sequential(*[
            Self3DAttentionToken(winsize, dim, channels)
            # Self3DAttentionToken_r(winsize, dim, channels)
            # Self3DAttentionToken_rh(winsize, dim, channels, 3)
            for i in range(layer)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, winsize, winsize, dim, 1))  # (1,15,15,36,1)
        self.pos_embed = nn.Parameter(torch.zeros(1, winsize, winsize, dim, channels + 1))  # (1,15,15,36,8)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1, -1, -1)
        x = torch.cat((cls_token, x), dim=-1)
        x = x + self.pos_embed
        x = self.encoders(x)
        return x[:, :, :, :, 1]
        # return x[:, :, :, :, 0]
    
# LK，spectral length：54，stride：0
class TensorTransformerHSI_unoverlapping_LK(nn.Module):
    def __init__(self):
        super(TensorTransformerHSI_unoverlapping_LK, self).__init__()
        self.encoder = TensorEncoder(7, 54, 5, 12)
        self.classifier = nn.Sequential(
            nn.Linear(54 * 7 * 7, 9),
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, 7, 7, 54, 5))
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
# LK，spectral length：54，stride：27
class TensorTransformerHSI_overlapping_LK(nn.Module):
    def __init__(self):
        super(TensorTransformerHSI_overlapping_LK, self).__init__()
        self.win = 7
        self.embed_dim = 54
        self.seq_num = 9
        self.encoder = TensorEncoder(self.win, self.embed_dim, self.seq_num, 12)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * self.win * self.win, 9),
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.win, self.win, 1, 270)) # (7,7,270) --> (7,7,1,270)
        step = int((x.shape[4] - self.embed_dim) / (self.seq_num - 1))  # stride = 27

        x_samples = torch.zeros(x.shape[0], self.win, self.win, self.seq_num, self.embed_dim).to('cuda:0')
        for i in range(self.seq_num):
            start_index = i * step
            end_index = start_index + self.embed_dim
            if end_index <= x.shape[-1]:  # 确保索引不会超出范围
                x_samples[..., i, :] = x[..., 0, start_index:end_index]
        x_samples = x_samples.permute(0, 1, 2, 4, 3)

        x = self.encoder(x_samples)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
class TensorTransformerHSI_unoverlapping_Hou(nn.Module):
    def __init__(self):
        super(TensorTransformerHSI_unoverlapping_Hou, self).__init__()
        self.win = 7
        self.encoder = TensorEncoder(self.win, 36, 4, 12)
        self.classifier = nn.Sequential(
            nn.Linear(36 * self.win * self.win, 15),
        )

    def forward(self, x):
        # (7,7,144) --> (7,7,36,4)
        x = torch.reshape(x, (-1, self.win, self.win, 36, 4))
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
# Hou： spectral length：54，stride：30
class TensorTransformerHSI_overlapping_Hou(nn.Module):
    def __init__(self):
        super(TensorTransformerHSI_overlapping_Hou, self).__init__()
        self.win = 7
        self.embed_dim = 54
        self.seq_num = 4
        self.encoder = TensorEncoder(self.win, self.embed_dim, self.seq_num, 12)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * self.win * self.win, 15),
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.win, self.win, 1, 144)) #(9,9,144) --> (9,9,1,144)
        step = int((x.shape[4] - self.embed_dim) / (self.seq_num - 1))

        x_samples = torch.zeros(x.shape[0], self.win, self.win, self.seq_num, self.embed_dim)#.to('cuda:0')
        for i in range(self.seq_num):
            start_index = i * step
            end_index = start_index + self.embed_dim
            if end_index <= x.shape[-1]:  # 确保索引不会超出范围
                x_samples[..., i, :] = x[..., 0, start_index:end_index]
        x_samples = x_samples.permute(0, 1, 2, 4, 3)

        x = self.encoder(x_samples)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    x = torch.randn(size=(8, 7, 7, 270))
    net = TensorTransformerHSI_unoverlapping_LK()
    # net = TensorTransformerHSI_overlapping_LK()

    # x = torch.randn(size=(8, 7, 7, 144))
    # net = TensorTransformerHSI_unoverlapping_Hou()
    # net = TensorTransformerHSI_overlapping_Hou()

    output = net(x)
>>>>>>> 28d2ef547ea47d584fd9708984d528f099f4b9da
    print(output.shape)