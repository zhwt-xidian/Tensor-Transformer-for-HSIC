<<<<<<< HEAD
'''
模型预测

'''

import os
import math
import argparse
import time
import torch
from torchvision import transforms
import importlib.util

import h5py
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from TensorTransformerFramework import TensorTransformerHSI_unoverlapping_Hou, TensorTransformerHSI_overlapping_Hou, TensorTransformerHSI_unoverlapping_LK, TensorTransformerHSI_overlapping_LK

# 加载配置
from config import label_path_dict, data_path_dict, parse

pyc_path = '__pycache__/utils.cpython-38.pyc'  
spec = importlib.util.spec_from_file_location("utils", pyc_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

global_config = parse.parse_args()


# 配置器
parse = argparse.ArgumentParser(description='Configuration of predict')
# 高斯滤波允许
parse.add_argument('--gaussian_filtering_enable', type=bool, default=True, metavar='N',help='enable/disable of gaussian filtering')
# 取窗大小
parse.add_argument('--apart_size', type=int, default=9, metavar='N', help='size of apart size')
# 数据准备模式（1：全图采样；0：非全图采样）
parse.add_argument('--dataprepare_mode', type=int, default=0, metavar='N', help='mode of dataprepare')
# 预测栈大小
parse.add_argument('--batch_size', type=int, default=256, metavar='N')
# padding模式（0：zero padding；1：ones padding）
parse.add_argument('--padding_mode', type=int, default=0, metavar='N', help='mode of padding')
# 数据集
parse.add_argument('--datasetName', type=str, default='SA', metavar='N', help='name of dataset')

# args = parse.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([transforms.ToTensor()])

label_path = label_path_dict[global_config.datasetName]
data_path = data_path_dict[global_config.datasetName]

data_label = scio.loadmat(label_path)
data_label = data_label[list(data_label.keys())[3]]
print('空间大小为：', data_label.shape)

# plt.imshow(data_label)
# plt.show()

# **************************************** 数据文件读取（带异常处理）*****************************************#

# data读取
try:
    data_set = scio.loadmat(data_path)
    data_set = data_set[list(data_set.keys())[3]]
except:
    data_set = h5py.File(data_path, 'r')
    data_set = np.transpose(data_set[list(data_set.keys())[0]])
    
# 数据预处理
if(global_config.gaussian_filtering_enable):
    
    print('数据预处理开始，使用高斯滤波')

    data_set = utils.image_Gaussian_conv(data_set)
    
    mean = np.mean(np.mean(data_set, axis=0), axis=0)
    std = np.std(np.std(data_set, axis=0), axis=0)
    data_set = (data_set - mean) / std
    
    data_set = utils.normalize(data_set)
    print('数据预处理结束')
    print("原始数据大小：", np.shape(data_set))

else:
    
    print('数据预处理开始，不使用高斯滤波')
    
    mean = np.mean(np.mean(data_set, axis=0), axis=0)
    std = np.std(np.std(data_set, axis=0), axis=0)
    data_set = (data_set - mean) / std
    
    data_set = utils.normalize(data_set)
    print('数据预处理结束')
    print("原始数据大小：", np.shape(data_set))
    
apart_size = global_config.sample_win_size
half_size = int((apart_size -1) / 2)

# if(args.dataprepare_mode == 1):
if(global_config.dataprepare_mode == 1):
    
    predict_label_area = data_label
    print('预测区域的大小是{}'.format(np.shape(predict_label_area)))
    scio.savemat('predict_label_area.mat',{'predict_label_area':predict_label_area})
    
    items = range(data_label.shape[0]*data_label.shape[1])
    
    padding_size = int((apart_size - 1) / 2)
    data_set = np.pad(data_set, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'constant', constant_values=global_config.padding_mode)
    data_label = np.pad(data_label, ((padding_size, padding_size), (padding_size, padding_size)), 'constant', constant_values=global_config.padding_mode)
    

else:
    
    x = data_set.shape[0]
    y = data_set.shape[1]
    z = data_set.shape[2]
    
    predict_label_area = data_label[half_size:x-half_size, half_size:y-half_size]
    print('预测区域的大小是{}'.format(np.shape(predict_label_area)))
    scio.savemat('predict_label_area.mat',{'predict_label_area':predict_label_area})
    
    items = range((x-apart_size)*(y-apart_size))
    
    plt.imshow(predict_label_area)
    plt.show()


# 模型选取
# model = TensorTransformerHSI_unoverlapping_LK().to(device)
model = TensorTransformerHSI_overlapping_LK().to(device)

# 加载权重
# weights_path = './TensorTransformerHSI_unoverlapping_LK.pth'
weights_path = './TensorTransformerHSI_overlapping_LK.pth'

assert os.path.exists(weights_path), "文件: '{}' 不存在".format(weights_path)
model.load_state_dict(torch.load(weights_path))

data_predict = np.zeros_like(predict_label_area)

batch_data = []
batch_label = []

batch_size = global_config.predict_batch_size

z_dim = data_set.shape[2]

predict_num = len(np.where(predict_label_area != 0)[0])
full_stack_num = math.floor(predict_num / batch_size)
stack_num_counter = 1
final_stack_size = predict_num - full_stack_num * batch_size

start = time.time()

with alive_bar(len(items)) as bar:
    
    for i in range(predict_label_area.shape[0]):  
        for j in range(predict_label_area.shape[1]):

            ii = i+half_size
            jj = j+half_size  #通过左上角位置
            
            if(predict_label_area[i][j] == 0):
                
                pass
            
            else:
                
                img = data_set[i:i+apart_size, j:j+apart_size, 0:z_dim]
                img = data_transform(img.astype(np.float32))
                
                # 一定要标准化！！！
                img_norm = utils.normalize(img)
                
                model.eval()
                    
                batch_data.append(img_norm)
                batch_label.append([i, j])

                if(stack_num_counter <= full_stack_num):
                    
                    if len(batch_data) == batch_size:
                        input_data = torch.stack(batch_data, dim=0)
                        input_data = input_data.type(torch.FloatTensor)

                        stack_num_counter = stack_num_counter + 1

                        model.eval()
                        with torch.no_grad():
                            
                            output = model(input_data.to(device))
                            pred = output.argmax(dim=1, keepdim=True)
                            
                            for index in range(pred.shape[0]):
                                index_x, index_y = batch_label[index]
                                data_predict[index_x, index_y] = pred.data[index, 0] +1
                            
                            batch_data.clear()
                            batch_label.clear()
                            
                else:
                    
                    if len(batch_data) == final_stack_size:
                        input_data = torch.stack(batch_data, dim=0)
                        input_data = input_data.type(torch.FloatTensor)
                        
                        stack_num_counter = stack_num_counter + 1

                        model.eval()
                        with torch.no_grad():
                            
                            output = model(input_data.to(device))
                            pred = output.argmax(dim=1, keepdim=True)
                            
                            for index in range(pred.shape[0]):
                                index_x, index_y = batch_label[index]
                                data_predict[index_x, index_y] = pred.data[index, 0] + 1
                            
                            batch_data.clear()
                            batch_label.clear()
    
            bar() 

# end = time.time()
# print(end-start)

# scio.savemat('TensorTransformerHSI_unoverlapping_LK_predict_result.mat',{'predict_result':data_predict})
scio.savemat('TensorTransformerHSI_overlapping_LK_predict_result.mat',{'predict_result':data_predict})
=======
'''
模型预测

'''

import os
import math
import argparse
import time
import torch
from torchvision import transforms
import importlib.util

import h5py
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from TensorTransformerFramework import TensorTransformerHSI_unoverlapping_Hou, TensorTransformerHSI_overlapping_Hou, TensorTransformerHSI_unoverlapping_LK, TensorTransformerHSI_overlapping_LK

# 加载配置
from config import label_path_dict, data_path_dict, parse

pyc_path = '__pycache__/utils.cpython-38.pyc'  
spec = importlib.util.spec_from_file_location("utils", pyc_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

global_config = parse.parse_args()


# 配置器
parse = argparse.ArgumentParser(description='Configuration of predict')
# 高斯滤波允许
parse.add_argument('--gaussian_filtering_enable', type=bool, default=True, metavar='N',help='enable/disable of gaussian filtering')
# 取窗大小
parse.add_argument('--apart_size', type=int, default=9, metavar='N', help='size of apart size')
# 数据准备模式（1：全图采样；0：非全图采样）
parse.add_argument('--dataprepare_mode', type=int, default=0, metavar='N', help='mode of dataprepare')
# 预测栈大小
parse.add_argument('--batch_size', type=int, default=256, metavar='N')
# padding模式（0：zero padding；1：ones padding）
parse.add_argument('--padding_mode', type=int, default=0, metavar='N', help='mode of padding')
# 数据集
parse.add_argument('--datasetName', type=str, default='SA', metavar='N', help='name of dataset')

# args = parse.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([transforms.ToTensor()])

label_path = label_path_dict[global_config.datasetName]
data_path = data_path_dict[global_config.datasetName]

data_label = scio.loadmat(label_path)
data_label = data_label[list(data_label.keys())[3]]
print('空间大小为：', data_label.shape)

# plt.imshow(data_label)
# plt.show()

# **************************************** 数据文件读取（带异常处理）*****************************************#

# data读取
try:
    data_set = scio.loadmat(data_path)
    data_set = data_set[list(data_set.keys())[3]]
except:
    data_set = h5py.File(data_path, 'r')
    data_set = np.transpose(data_set[list(data_set.keys())[0]])
    
# 数据预处理
if(global_config.gaussian_filtering_enable):
    
    print('数据预处理开始，使用高斯滤波')

    data_set = utils.image_Gaussian_conv(data_set)
    
    mean = np.mean(np.mean(data_set, axis=0), axis=0)
    std = np.std(np.std(data_set, axis=0), axis=0)
    data_set = (data_set - mean) / std
    
    data_set = utils.normalize(data_set)
    print('数据预处理结束')
    print("原始数据大小：", np.shape(data_set))

else:
    
    print('数据预处理开始，不使用高斯滤波')
    
    mean = np.mean(np.mean(data_set, axis=0), axis=0)
    std = np.std(np.std(data_set, axis=0), axis=0)
    data_set = (data_set - mean) / std
    
    data_set = utils.normalize(data_set)
    print('数据预处理结束')
    print("原始数据大小：", np.shape(data_set))
    
apart_size = global_config.sample_win_size
half_size = int((apart_size -1) / 2)

# if(args.dataprepare_mode == 1):
if(global_config.dataprepare_mode == 1):
    
    predict_label_area = data_label
    print('预测区域的大小是{}'.format(np.shape(predict_label_area)))
    scio.savemat('predict_label_area.mat',{'predict_label_area':predict_label_area})
    
    items = range(data_label.shape[0]*data_label.shape[1])
    
    padding_size = int((apart_size - 1) / 2)
    data_set = np.pad(data_set, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'constant', constant_values=global_config.padding_mode)
    data_label = np.pad(data_label, ((padding_size, padding_size), (padding_size, padding_size)), 'constant', constant_values=global_config.padding_mode)
    

else:
    
    x = data_set.shape[0]
    y = data_set.shape[1]
    z = data_set.shape[2]
    
    predict_label_area = data_label[half_size:x-half_size, half_size:y-half_size]
    print('预测区域的大小是{}'.format(np.shape(predict_label_area)))
    scio.savemat('predict_label_area.mat',{'predict_label_area':predict_label_area})
    
    items = range((x-apart_size)*(y-apart_size))
    
    plt.imshow(predict_label_area)
    plt.show()


# 模型选取
# model = TensorTransformerHSI_unoverlapping_LK().to(device)
model = TensorTransformerHSI_overlapping_LK().to(device)

# 加载权重
# weights_path = './TensorTransformerHSI_unoverlapping_LK.pth'
weights_path = './TensorTransformerHSI_overlapping_LK.pth'

assert os.path.exists(weights_path), "文件: '{}' 不存在".format(weights_path)
model.load_state_dict(torch.load(weights_path))

data_predict = np.zeros_like(predict_label_area)

batch_data = []
batch_label = []

batch_size = global_config.predict_batch_size

z_dim = data_set.shape[2]

predict_num = len(np.where(predict_label_area != 0)[0])
full_stack_num = math.floor(predict_num / batch_size)
stack_num_counter = 1
final_stack_size = predict_num - full_stack_num * batch_size

start = time.time()

with alive_bar(len(items)) as bar:
    
    for i in range(predict_label_area.shape[0]):  
        for j in range(predict_label_area.shape[1]):

            ii = i+half_size
            jj = j+half_size  #通过左上角位置
            
            if(predict_label_area[i][j] == 0):
                
                pass
            
            else:
                
                img = data_set[i:i+apart_size, j:j+apart_size, 0:z_dim]
                img = data_transform(img.astype(np.float32))
                
                # 一定要标准化！！！
                img_norm = utils.normalize(img)
                
                model.eval()
                    
                batch_data.append(img_norm)
                batch_label.append([i, j])

                if(stack_num_counter <= full_stack_num):
                    
                    if len(batch_data) == batch_size:
                        input_data = torch.stack(batch_data, dim=0)
                        input_data = input_data.type(torch.FloatTensor)

                        stack_num_counter = stack_num_counter + 1

                        model.eval()
                        with torch.no_grad():
                            
                            output = model(input_data.to(device))
                            pred = output.argmax(dim=1, keepdim=True)
                            
                            for index in range(pred.shape[0]):
                                index_x, index_y = batch_label[index]
                                data_predict[index_x, index_y] = pred.data[index, 0] +1
                            
                            batch_data.clear()
                            batch_label.clear()
                            
                else:
                    
                    if len(batch_data) == final_stack_size:
                        input_data = torch.stack(batch_data, dim=0)
                        input_data = input_data.type(torch.FloatTensor)
                        
                        stack_num_counter = stack_num_counter + 1

                        model.eval()
                        with torch.no_grad():
                            
                            output = model(input_data.to(device))
                            pred = output.argmax(dim=1, keepdim=True)
                            
                            for index in range(pred.shape[0]):
                                index_x, index_y = batch_label[index]
                                data_predict[index_x, index_y] = pred.data[index, 0] + 1
                            
                            batch_data.clear()
                            batch_label.clear()
    
            bar() 

# end = time.time()
# print(end-start)

# scio.savemat('TensorTransformerHSI_unoverlapping_LK_predict_result.mat',{'predict_result':data_predict})
scio.savemat('TensorTransformerHSI_overlapping_LK_predict_result.mat',{'predict_result':data_predict})
>>>>>>> 28d2ef547ea47d584fd9708984d528f099f4b9da
