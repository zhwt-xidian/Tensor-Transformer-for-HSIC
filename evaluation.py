<<<<<<< HEAD
'''
评估预测结果

'''

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import seaborn as sns
import importlib.util
pyc_path = '__pycache__/utils.cpython-38.pyc'  
spec = importlib.util.spec_from_file_location("utils", pyc_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
import copy
from config import label_path_dict, data_path_dict, parse

global_config = parse.parse_args()

# 读取预测文件
# data_predict = scio.loadmat('./TensorTransformerHSI_unoverlapping_LK_predict_result.mat')
data_predict = scio.loadmat('./TensorTransformerHSI_overlapping_LK_predict_result.mat')
data_predict = data_predict[list(data_predict.keys())[3]]

data_raw = copy.deepcopy(data_predict)
predict_label = scio.loadmat('./predict_label_area.mat')
predict_label = predict_label[list(predict_label.keys())[3]]

data_predict = utils.ClassficationResultFilterd(data_predict, 7)
data_predict_show = copy.deepcopy(data_predict)

# 保存预测结果图
# scio.savemat("TensorTransformerHSI_unoverlapping_LK_predict_result_filter.mat", { 'predict_result_filter' : data_predict })
scio.savemat("TensorTransformerHSI_overlapping_LK_predict_result_filter.mat", { 'predict_result_filter' : data_predict })

label = predict_label
data_predict = np.reshape(data_predict, (-1, ))
label = np.reshape(label, (-1, ))

index = np.where(label == 0)

label = np.delete(label, index)
data_predict = np.delete(data_predict, index)

overroll_acc = accuracy_score(label, data_predict)
average_accuracy = utils.average_accuracy_score(label, data_predict)
Kappa = cohen_kappa_score(label, data_predict)

print("OA: {}".format(overroll_acc))
print("AA: {}".format(average_accuracy))
print("Kappa: {}".format(Kappa))

scio.savemat("metrics.mat", {'OA' : overroll_acc, 'Kappa' : Kappa, 'AA' : average_accuracy})

plt.subplot(121)
plt.imshow(data_raw)
plt.subplot(122)
plt.imshow(data_predict_show)
plt.show()

print("\n" + "混淆矩阵如下")
print(confusion_matrix(label, data_predict))
utils.plot_confusion_matrix(confusion_matrix(label, data_predict))

# 每类正确率
cm = confusion_matrix(label, data_predict)
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i in range(9):
=======
'''
评估预测结果

'''

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import seaborn as sns
import importlib.util
pyc_path = '__pycache__/utils.cpython-38.pyc'  
spec = importlib.util.spec_from_file_location("utils", pyc_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
import copy
from config import label_path_dict, data_path_dict, parse

global_config = parse.parse_args()

# 读取预测文件
# data_predict = scio.loadmat('./TensorTransformerHSI_unoverlapping_LK_predict_result.mat')
data_predict = scio.loadmat('./TensorTransformerHSI_overlapping_LK_predict_result.mat')
data_predict = data_predict[list(data_predict.keys())[3]]

data_raw = copy.deepcopy(data_predict)
predict_label = scio.loadmat('./predict_label_area.mat')
predict_label = predict_label[list(predict_label.keys())[3]]

data_predict = utils.ClassficationResultFilterd(data_predict, 7)
data_predict_show = copy.deepcopy(data_predict)

# 保存预测结果图
# scio.savemat("TensorTransformerHSI_unoverlapping_LK_predict_result_filter.mat", { 'predict_result_filter' : data_predict })
scio.savemat("TensorTransformerHSI_overlapping_LK_predict_result_filter.mat", { 'predict_result_filter' : data_predict })

label = predict_label
data_predict = np.reshape(data_predict, (-1, ))
label = np.reshape(label, (-1, ))

index = np.where(label == 0)

label = np.delete(label, index)
data_predict = np.delete(data_predict, index)

overroll_acc = accuracy_score(label, data_predict)
average_accuracy = utils.average_accuracy_score(label, data_predict)
Kappa = cohen_kappa_score(label, data_predict)

print("OA: {}".format(overroll_acc))
print("AA: {}".format(average_accuracy))
print("Kappa: {}".format(Kappa))

scio.savemat("metrics.mat", {'OA' : overroll_acc, 'Kappa' : Kappa, 'AA' : average_accuracy})

plt.subplot(121)
plt.imshow(data_raw)
plt.subplot(122)
plt.imshow(data_predict_show)
plt.show()

print("\n" + "混淆矩阵如下")
print(confusion_matrix(label, data_predict))
utils.plot_confusion_matrix(confusion_matrix(label, data_predict))

# 每类正确率
cm = confusion_matrix(label, data_predict)
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i in range(9):
>>>>>>> 28d2ef547ea47d584fd9708984d528f099f4b9da
    print(f"Class {i+1} accuracy: {class_accuracy[i]}")