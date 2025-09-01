<<<<<<< HEAD
'''
全局配置文件

'''

import argparse

parse = argparse.ArgumentParser('Tensor Classification Network Arguement Parser')

label_path_dict = { 'PU' : './Dataset/pavia_uni_gt.mat', 
                    'SA' : './Dataset/Salinas_gt.mat',
                    'LongKou' : './Dataset/WHU_Hi_LongKou_gt.mat', 
                    'HanChuan' : './Dataset/WHU_Hi_HanChuan_gt.mat', 
                    'HongHu' : './Dataset/WHU_Hi_HongHu_gt.mat',
                    'Indian' : "./Dataset/Indian_pines_gt.mat",
                    'Indian_Head' : './Dataset/indian_head_farm_label.mat',
                    'Houston' : './Dataset/Houston_gt.mat'
                    }

data_path_dict = { 'PU' : './Dataset/pavia_uni.mat', 
                   'SA' : './Dataset/Salinas_corrected.mat', 
                   'LongKou' : './Dataset/WHU_Hi_LongKou.mat',
                   'HanChuan' : './Dataset/WHU_Hi_HanChuan.mat',
                   'HongHu' : './Dataset/WHU_Hi_HongHu.mat',
                   'Indian' : "./Dataset/Indian_pines_corrected.mat",
                   'Indian_Head' : "./Dataset/indian_head_farm.mat",
                   'Houston' : './Dataset/Houston2013.mat'
                   }

# 公共配置
# 数据集
parse.add_argument('--datasetName', type=str, default='LongKou', metavar='N', help='name of dataset')
# 高斯滤波允许
parse.add_argument('--gaussian_filtering_enable', type=bool, default=True, metavar='N',help='enable/disable of gaussian filtering')
# 取窗大小
parse.add_argument('--sample_win_size', type=int, default=7, metavar='N', help='size of apart size')
# 数据准备padding模式（0：zero padding；1：ones padding）
parse.add_argument('--padding_mode', type=int, default=0, metavar='N', help='mode of padding')
# 数据保存位置 
parse.add_argument('--data_save_path', type=str, default='./Dataset/Data/', metavar='N', help='path of data save path')
# 数据准备模式（1：全图采样；0：非全图采样）
parse.add_argument('--dataprepare_mode', type=int, default=1, metavar='N', help='mode of dataprepare')

# 数据准备配置
# 训练集采样率
parse.add_argument('--train_data_rate', type=float, default=0.005, metavar='N', help='rate of train data')
# 验证集采样率
parse.add_argument('--val_data_rate', type=float, default=0.0025, metavar='N', help='rate of train data')
# 数据取样模式（0：类等比率取样；1：类等值取样）
parse.add_argument('--sampling_mode', type=int, default=1, metavar='N', help='mode of sampling')
# 等值采样模式采样数
parse.add_argument('--sampling_num', type=tuple, default=(20, 10), metavar="N", help='sampling number')

# 训练配置
# 训练batch大小
parse.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
# 训练轮数
parse.add_argument('--epochs', type=int, default=500, metavar='N', help='epochs for training (default: 500)')
# 模型文件保存位置
parse.add_argument('--model_save_path', type=str, default='./TNN.pth', help='the path to save model')
# 是否打开训练监控界面
parse.add_argument('--train_monitor', type=bool, default=True, help='open train monitor gui')

# 预测配置
# 预测栈大小
parse.add_argument('--predict_batch_size', type=int, default=512, metavar='N')
=======
'''
全局配置文件

'''

import argparse

parse = argparse.ArgumentParser('Tensor Classification Network Arguement Parser')

label_path_dict = { 'PU' : './Dataset/pavia_uni_gt.mat', 
                    'SA' : './Dataset/Salinas_gt.mat',
                    'LongKou' : './Dataset/WHU_Hi_LongKou_gt.mat', 
                    'HanChuan' : './Dataset/WHU_Hi_HanChuan_gt.mat', 
                    'HongHu' : './Dataset/WHU_Hi_HongHu_gt.mat',
                    'Indian' : "./Dataset/Indian_pines_gt.mat",
                    'Indian_Head' : './Dataset/indian_head_farm_label.mat',
                    'Houston' : './Dataset/Houston_gt.mat'
                    }

data_path_dict = { 'PU' : './Dataset/pavia_uni.mat', 
                   'SA' : './Dataset/Salinas_corrected.mat', 
                   'LongKou' : './Dataset/WHU_Hi_LongKou.mat',
                   'HanChuan' : './Dataset/WHU_Hi_HanChuan.mat',
                   'HongHu' : './Dataset/WHU_Hi_HongHu.mat',
                   'Indian' : "./Dataset/Indian_pines_corrected.mat",
                   'Indian_Head' : "./Dataset/indian_head_farm.mat",
                   'Houston' : './Dataset/Houston2013.mat'
                   }

# 公共配置
# 数据集
parse.add_argument('--datasetName', type=str, default='LongKou', metavar='N', help='name of dataset')
# 高斯滤波允许
parse.add_argument('--gaussian_filtering_enable', type=bool, default=True, metavar='N',help='enable/disable of gaussian filtering')
# 取窗大小
parse.add_argument('--sample_win_size', type=int, default=7, metavar='N', help='size of apart size')
# 数据准备padding模式（0：zero padding；1：ones padding）
parse.add_argument('--padding_mode', type=int, default=0, metavar='N', help='mode of padding')
# 数据保存位置 
parse.add_argument('--data_save_path', type=str, default='./Dataset/Data/', metavar='N', help='path of data save path')
# 数据准备模式（1：全图采样；0：非全图采样）
parse.add_argument('--dataprepare_mode', type=int, default=1, metavar='N', help='mode of dataprepare')

# 数据准备配置
# 训练集采样率
parse.add_argument('--train_data_rate', type=float, default=0.005, metavar='N', help='rate of train data')
# 验证集采样率
parse.add_argument('--val_data_rate', type=float, default=0.0025, metavar='N', help='rate of train data')
# 数据取样模式（0：类等比率取样；1：类等值取样）
parse.add_argument('--sampling_mode', type=int, default=1, metavar='N', help='mode of sampling')
# 等值采样模式采样数
parse.add_argument('--sampling_num', type=tuple, default=(20, 10), metavar="N", help='sampling number')

# 训练配置
# 训练batch大小
parse.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
# 训练轮数
parse.add_argument('--epochs', type=int, default=500, metavar='N', help='epochs for training (default: 500)')
# 模型文件保存位置
parse.add_argument('--model_save_path', type=str, default='./TNN.pth', help='the path to save model')
# 是否打开训练监控界面
parse.add_argument('--train_monitor', type=bool, default=True, help='open train monitor gui')

# 预测配置
# 预测栈大小
parse.add_argument('--predict_batch_size', type=int, default=512, metavar='N')
>>>>>>> 28d2ef547ea47d584fd9708984d528f099f4b9da
