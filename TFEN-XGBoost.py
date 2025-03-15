#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pandas
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, \
    Activation, GlobalAveragePooling1D, DepthwiseConv2D, Add, GRU, LSTM, Permute, multiply, \
    MaxPool1D, Bidirectional
from tensorflow.keras.layers import add, Flatten, Reshape, ReLU, concatenate, SeparableConv1D
from tensorflow.keras import regularizers

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn import metrics

# mat
import scipy.io

# Plot
import matplotlib.pyplot as plt

# Utility
import os
import glob
import numpy as np
from tqdm import tqdm
import itertools
import pickle
import sklearn
from scipy import signal

os.environ['KERAS_BACKEND']='tensorflow'

import sys
import scipy
from scipy import ndimage
import tensorflow as tf
from PIL import Image
import random

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# In[2]:

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[3]:

conf_path = os.path.dirname(os.path.abspath('__file__'))
work_path = os.path.dirname(os.path.dirname(conf_path))
# 打印work_path的路径
#print("Work path:", work_path)
setd_path = os.path.join(work_path, "feature extraction","havesj_4", "**")

# In[4]:

import time

start_time = time.time()

dataset = []
for folder in [setd_path]:
    # for folder in [data_path]:
    for filename in glob.iglob(folder):
        if os.path.exists(filename):
            label = os.path.basename(filename).split("-")[0]
            # print("$$$$$$$$:",len(label))
            dataset.append({"filename": filename, "label": label})
dataset = pd.DataFrame(dataset)
# 定义自定义标签排序顺序
label_order = ['LL', 'LH', 'HL', 'HH']
# 将 'label' 列转换为分类类型并指定顺序
dataset['label'] = pd.Categorical(dataset['label'], categories=label_order, ordered=True)

dataset = shuffle(dataset, random_state=42)
dataset.info()  # 查看信息

end_time = time.time()
execution_time = end_time - start_time
print("Execution time: ", execution_time)

# In[5]:

train, test = train_test_split(dataset, test_size=0.1, random_state=20)
train, validation = train_test_split(dataset, test_size=0.2, random_state=20)

print("Train: %i" % len(train))
print("Validation: %i" % len(validation))
print("Test: %i" % len(test))

print(dataset.label.unique())
# classes = dataset.label.unique()
classes = dataset['label'].cat.categories
print("####:",classes)
print("!!!!:",len(classes))
# In[6]:

def extract_features(data_path):
    y =scipy.io.loadmat(data_path)
    # 检查字典 y 中是否存在键 'Exlog'
    if 'xn_suijilog' in y.keys():
        # 如果存在，将 y['Exlog'] 的值赋给变量 data
        data = y['xn_suijilog']
    else:
        # 如果不存在，可以选择抛出异常或者返回 None
        # 这里选择抛出 ValueError 异常
        raise ValueError(f"The key 'Exlog' does not exist in the file: {data_path}")
    # 返回提取的数据
    return data

x_test,x_train,x_validation = [],[],[]

print("Extract features from TRAIN,VALIDATION and TEST dataset")

for idx in tqdm(range(len(test))):
    x_test.append(extract_features(test.filename.iloc[idx]))
    
for idx in tqdm(range(len(train))):
    x_train.append(extract_features(train.filename.iloc[idx]))
    
for idx in tqdm(range(len(validation))):
    x_validation.append(extract_features(validation.filename.iloc[idx]))

x_test = np.array(x_test)
print("X test:", x_test.shape)

x_train = np.array(x_train)
print("X train:", x_train.shape)

x_validation = np.array(x_validation)
print("X validation:", x_validation.shape)

# In[7]:

start_time = time.time()
encoder = LabelEncoder()
encoder.fit(train['label'])

y_train = encoder.transform(train['label'])
y_validation = encoder.transform(validation['label'])
y_test = encoder.transform(test['label'])

print(y_train.shape)
print(y_validation.shape)
print(y_test.shape)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# In[8]:
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

xb_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
xb_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1])
xb_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
yb_train=y_train
yb_validation=y_validation
yb_test=y_test


# In[9]:


y_train = to_categorical(y_train)
y_validation = to_categorical(y_validation)
y_test = to_categorical(y_test)


# In[10]:


class ResNet_GRU_build_model:
    def __init__(self, trainData, trainLabel, validationData, validationLabel):
        self.trainData = trainData
        self.trainLabel = trainLabel
        self.batch_size = 512  # 批大小
        self.epochs = 300
        self.verbose = 1
        self.num_class = len(classes)   # 分类数目
        self.optimizer = tf.keras.optimizers.RMSprop(lr=0.001)
        self.validation = (validationData, validationLabel)
        self.width = 180
        self.height = 1
        self.channel = 1
        self.stride = 2

    def conv1d_bn(self, x, nb_filter, padding='same'):
        """
        conv1d -> batch normalization -> relu activation
        """
        x = Conv1D(nb_filter, kernel_size=3,
                   strides=1,
                   padding=padding,
                   dilation_rate=1,
                   kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def shortcut(self, input, residual):
        """
        shortcut连接，也就是identity mapping部分。
        """

        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride = int(round(input_shape[1] / residual_shape[1]))
        equal_channels = input_shape[2] == residual_shape[2]

        identity = input
        # 如果维度不同，则使用1x1卷积进行调整
        if stride > 1 or not equal_channels:
            identity = Conv1D(filters=residual_shape[2],
                              kernel_size=1,
                              strides=stride,
                              padding="valid",
                              kernel_regularizer=regularizers.l2(0.001))(input)

        return add([identity, residual])

    def basic_block(self, nb_filter, strides=1):
        """
        基本的ResNet building block，适用于ResNet-18和ResNet-34.
        """

        def f(input):
            conv1 = self.conv1d_bn(input, nb_filter)
            residual = self.conv1d_bn(conv1, nb_filter)

            return self.shortcut(input, residual)

        return f

    def residual_block(self, nb_filter, repetitions, is_first_layer=False):
        """
        构建每层的residual模块，对应论文参数统计表中的conv2_x -> conv5_x
        """

        def f(input):
            for i in range(repetitions):
                strides = 1
                if i == 0 and not is_first_layer:
                    strides = 2
                input = self.basic_block(nb_filter, strides)(input)
            return input

        return f
    def conv1d_bn1(self, x, nb_filter, padding='same'):
        """
        dilated_conv1d -> batch normalization -> relu activation
        """
        x = Conv1D(nb_filter, kernel_size=3,
                   strides=1,
                   padding=padding,
                   dilation_rate=2,
                   kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def basic_block1(self, nb_filter, strides=1):
        """
        基本的ResNet building block，适用于ResNet-18和ResNet-34.
        """

        def f(input):
            conv1 = self.conv1d_bn1(input, nb_filter)
            residual = self.conv1d_bn1(conv1, nb_filter)

            return self.shortcut(input, residual)

        return f

    def residual_block1(self, nb_filter, repetitions, is_first_layer=False):
        """
        构建每层的residual模块，对应论文参数统计表中的conv2_x -> conv5_x
        """

        def f(input):
            for i in range(repetitions):
                strides = 1
                if i == 0 and not is_first_layer:
                    strides = 2
                input = self.basic_block1(nb_filter, strides)(input)
            return input

        return f

    def ResNet_GRU_build_model(self):

        inputs = Input(shape=(self.width, self.height))

        conv1 = Conv1D(strides=2, kernel_size=1, filters=96, padding='same', activation='relu')(inputs)
        pool1 = MaxPool1D(pool_size=2, strides=2)(conv1)
        print(pool1.shape)

        conv2 = self.conv1d_bn(pool1, 64)
        pool2 = MaxPool1D(pool_size=2, padding='same')(conv2)

        conv3 = self.residual_block(64, 1, is_first_layer=True)(pool2)

        conv4 = self.residual_block1(128, 1, is_first_layer=True)(conv3)
        dropout1 = Dropout(0.5)(conv4)
        # conv5 = self.residual_block(128, 1, is_first_layer=True)(conv3)
        # dropout2 = Dropout(0.5)(conv5)
        # conv5 = self.residual_block(512, 1, is_first_layer=True)(dropout2)
        conv6 = Conv1D(strides=2, kernel_size=3, filters=128, padding='same', activation='relu')(dropout1)
        pool6 = MaxPool1D(pool_size=2, strides=2)(conv6)
        # conv7 = Conv1D(strides=2, kernel_size=1, filters=256, padding='same', activation='relu')(pool6)
        # pool7 = MaxPool1D(pool_size=2, strides=2)(conv1)

        gap = GlobalAveragePooling1D()(pool6)
        x = Flatten()(gap)
        y = LSTM(units=50, return_sequences=True)(pool1)
        y = LSTM(units=20, return_sequences=True)(y)
        y = Flatten()(y)
        xy = concatenate([x, y], axis=1)
        print("!!!!:", xy.shape)
        # output = Dense(10,kernel_regularizer=regularizers.l2(0.1))(xy)
        output = Dense(self.num_class, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1))(xy)
        model = Model(inputs, output)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def ResNet_GRU_train_model(self):
        model = self.ResNet_GRU_build_model()

        def scheduler(epoch):
            # 每隔100个epoch，学习率减小为原来的1/10
            if epoch % 50 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.1)
                print("lr changed to {}".format(lr * 0.1))
            return K.get_value(model.optimizer.lr)

        reduce_lr = LearningRateScheduler(scheduler)
        print('ResNet_GRU: ', model.summary())
        modelInfo = model.fit(self.trainData, self.trainLabel, batch_size=self.batch_size,
                              validation_data=self.validation, verbose=self.verbose, shuffle=True, epochs=self.epochs,
                              callbacks=[reduce_lr])
        return model, modelInfo


# In[12]:


model = ResNet_GRU_build_model(x_train, y_train, x_validation, y_validation)
g_mode, g_info = model.ResNet_GRU_train_model()


# In[14]:


# Loss Curves
plt.figure(figsize=[16,12])
plt.subplot(211)
plt.plot(g_info.history['loss'],'r',linewidth=3.0)
#plt.plot(g_info.history['val_loss'],'b',linestyle=':',linewidth=3.0)
plt.plot(g_info.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

# Accuracy Curves
plt.subplot(212)
plt.plot(g_info.history['accuracy'],'r',linewidth=3.0)
#plt.plot(g_info.history['val_accuracy'],'b',linestyle=':',linewidth=3.0)
plt.plot(g_info.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
# 调整子图间距
plt.subplots_adjust(hspace=0.6)  # 调整子图之间的垂直间距
plt.savefig(r'E:\feature extraction\\b1sj\Accuracy_curves.png')
# In[15]:

start_time = time.time()

g_score1 = g_mode.evaluate(x_test, y_test, verbose=1)
print('Test loss:', g_score1[0])
print('Test accuracy:', g_score1[1])

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# In[17]:

from tensorflow.keras.models import Model
layer_name = 'concatenate'
# layer_name='flatten'
intermediate_layer_model = Model(inputs=g_mode.input,
                                 outputs=g_mode.get_layer(layer_name).output)

#intermediate_layer_model.summary()

# 指定要保存摘要信息的文件夹和文件名
folder_path = 'E:\\feature extraction\\b1sj'  # 指定文件夹路径
file_name = 'model_summary.txt'  # 指定文件名
file_path = os.path.join(folder_path, file_name)  # 使用 os.path.join 确保路径正确
# 确保文件夹存在
os.makedirs(folder_path, exist_ok=True)

# 保存模型摘要到文本文件
with open(file_path, 'w') as f:
    intermediate_layer_model.summary(print_fn=lambda x: f.write(x + '\n'))

# In[18]:


intermediate_output = intermediate_layer_model.predict(x_train)
intermediate_output = pd.DataFrame(data=intermediate_output)
intermediate_test_output = intermediate_layer_model.predict(x_test)
intermediate_test_output = pd.DataFrame(data=intermediate_test_output)


# In[19]:

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets  import  make_hastie_10_2
from xgboost.sklearn import XGBClassifier
#
#
# In[20]:

# clf = XGBClassifier(
# silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
# #nthread=4,# cpu 线程数 默认最大
# learning_rate= 0.1, # 如同学习率
# min_child_weight=3.5,
# # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
# #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
# #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
# max_depth=6, # 构建树的深度，越大越容易过拟合
# gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
# subsample=1, # 随机采样训练样本 训练实例的子采样比
# max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
# colsample_bytree=1, # 生成树时进行的列采样
# reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
# #reg_alpha=0, # L1 正则项参数
# #scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
# #objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
# #num_class=10, # 类别数，多分类与 multisoftmax 并用
# n_estimators=100, #树的个数
# seed=1000 #随机种子
# #eval_metric= 'auc'
#  )

from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
from xgboost import to_graphviz
# 画图-混淆矩阵
import seaborn as sns
from sklearn.metrics import confusion_matrix
# ROCq曲线AUC值
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
# 用来正常显示负号
from matplotlib.ticker import FuncFormatter

def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'

xgb = XGBClassifier()
# scale_pos_weight,正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。
# 建立需要搜索的参数的范围

param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 6, 7], 'learning_rate': [0.01, 0.1, 0.2]}
grid_search_xgb = GridSearchCV(xgb, param_grid, cv=3)
try:
    clf = grid_search_xgb.fit(intermediate_output, yb_train)
except:
    clf = grid_search_xgb.fit(intermediate_output, yb_train)

# clf.fit(intermediate_output, yb_train)
# 获取最优模型
best_model = grid_search_xgb.best_estimator_

# # ********************************************************************************************保存xgb网格参数
outPath = r'E:\feature extraction\b1sj'
f = open(outPath + '\\' + r'_xgbParameter.txt', "w")
f.write("xgb参数说明：\n")
f.write('搜索网格参数2.树的深度(max_depth)\n')
f.write('\t[5,6,7]值过大容易过拟合，值过小容易欠拟合\n')
f.write('搜索网格参数3.基分类器的个数(n_estimatores)\n')
f.write('\t[80, 100, 120]\n')
f.write('搜索最优网格参数:' + '\n' + str(grid_search_xgb.best_params_) + '\n')
f.write('默认超参数:' + "\n" + str(best_model) + '\n')
f.close()
# # ********************************************************************************************

# In[23]:

start_time = time.time()

print(clf.score(intermediate_output, yb_train))
print(clf.score(intermediate_test_output, yb_test))

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# In[22]:
y_scoreXGB1 = clf.predict(intermediate_test_output)
# 预测标签
y_scoreXGB = clf.predict_proba(intermediate_test_output)

# 预测标签概率
print('预测概率：', y_scoreXGB)
# print(type(y_scoreXGB))
print("##:", y_scoreXGB.shape)
print("??:", yb_test.shape)

y_scoreXGB_df = pd.DataFrame(y_scoreXGB)
yb_test_df = pd.DataFrame(yb_test)

y_scoreXGB_df.to_csv('y_scoreXGB.csv', index=False)
yb_test_df.to_csv('yb_test.csv', index=False)

# 假设 y_scoreXGB 是预测的概率，shape 为 (n_samples, 4)
# 假设 testTarget 是实际的目标值，shape 为 (n_samples,)

# 获取每个样本最可能的类别标签
y_pred = np.argmax(y_scoreXGB, axis=1)

# 找出预测错误的样本索引
error_indices = np.where(y_pred != yb_test)[0]
print(error_indices)
# 创建一个包含错误预测的 DataFrame
error_df = pd.DataFrame({
    'Predicted': y_pred[error_indices],
    'Actual': yb_test[error_indices]
})

# 打印错误预测
print("预测错误的数据:")
print(error_df)

# 如果你想要将这些错误预测保存到 CSV 文件
error_df.to_csv('prediction_errors.csv', index=False)

accuracy = clf.score(intermediate_test_output, yb_test)
misclass = 1 - accuracy
# 打印准确率
print("xgb准确率：", accuracy)

# 计算精确率
precision = precision_score(yb_test, y_scoreXGB1, average='macro')

# 计算召回率
recall = recall_score(yb_test, y_scoreXGB1, average='macro')

# 计算F1-Score
f1 = f1_score(yb_test, y_scoreXGB1, average='macro')
print("xgb精确率:", precision)
print("xgb召回率:", recall)
print("xgbF1-Score:", f1)
# 指定保存结果的文件夹路径
save_folder = 'E:/feature extraction/b1sj'
# 保存结果到文件
results = f"准确率: {accuracy:.4f}\n精确率: {precision:.4f}\n召回率: {recall:.4f}\nF1-Score: {f1:.4f}"
with open(os.path.join(save_folder, 'xgb_results.txt'), 'w') as f:
    f.write(results)

print('Train accuracy:', clf.score(intermediate_output, yb_train))
print('Test accuracy:', clf.score(intermediate_test_output, yb_test))

testTarget1 = label_binarize(yb_test, np.arange(len(classes)))


# In[26]:

# plt.plot(xb_train[1])

# **************************************************************************************************************xgb混淆矩阵
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# # 假设classes是当前的类别顺序
# current_classes = np.unique(yb_test)
# target_order = ['LL', 'LH', 'HL', 'HH']
#
# # 创建LabelEncoder实例
# le = LabelEncoder()
# # 拟合目标顺序
# le.fit(target_order)
#
# # 使用LabelEncoder将yb_test和预测结果转换为数值标签
# yb_test_encoded = le.transform(yb_test)
# y_pred_encoded = le.transform(clf.predict(intermediate_test_output))

matrixFindsize = 12
# 混淆矩阵字体大小
matrixDip = 120
# 分辨率
# Plot confusion matrix  绘图混淆矩阵
sns.set_context("talk", rc={"font": "Helvetica", "font.size": matrixFindsize})
#label = [files[i] for i in classOneLabel]
cm = confusion_matrix(yb_test, clf.predict(intermediate_test_output))
# cm = confusion_matrix(yb_test_encoded, y_pred_encoded, labels=le.classes_)
plt.figure(figsize=(8, 6),dpi=120,facecolor='blueviolet')
# 设置title
plt.title('Test results of xgb classifier', fontsize=12)

cm = 100 * cm / cm.sum(axis=1)[:, np.newaxis]
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='0.1f', xticklabels=classes, yticklabels=classes)
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.xticks(rotation=90)
plt.yticks(rotation=0)
# 设置刻度值即坐标轴刻度 xticks yticks
plt.tick_params(labelsize=10)
plt.xlabel('Predicted_label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=12)
plt.ylabel('True_label', fontsize=12)
# 设置坐标轴的刻度参数

outPath1 = outPath + '\\' + r'_XGB.tif'
plt.savefig(outPath1, format='tif', bbox_inches='tight', transparent=True, dpi=300)
# outPath122 = outPath + '\\' + r'_XGB.pdf'
# plt.savefig(outPath122, format='pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
# *************************************************************************************XGB的ROC曲线与AUC面积
print("!!!!:", testTarget1.shape)
print("###:", y_scoreXGB.shape)
if 2 == len(classes):
    # 二分类ROC曲线
    fpr, tpr, threshold = roc_curve(yb_test, y_scoreXGB1)
    ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)
    ###计算auc的值
    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('XGB_ROC', fontsize=12)
    plt.legend(loc="lower right", fontsize=8)
    # plt.show()
    outPath121 = outPath + '\\' + r'_XGB_ROC.tif'
    plt.savefig(outPath121, format='tif', bbox_inches='tight', transparent=True, dpi=300)
    plt.rcParams['svg.fonttype'] = 'none'
    outPath121 = outPath + '\\' + r'_XGB_ROC.svg'
    plt.savefig(outPath121, format='svg', bbox_inches='tight', transparent=True)
    plt.close()
else:
    # 多分类
    # 初始化 fpr 和 tpr
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(testTarget1[:, i], y_scoreXGB[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(testTarget1.ravel(), y_scoreXGB.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AU
    mean_tpr /= len(classes)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':')

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':')

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('XGB_ROC:multi-class', fontsize=12)
    plt.legend(loc="lower right", fontsize=8)
    # plt.show()
    outPath121 = outPath + '\\' + r'_XGB_ROC.tif'
    plt.savefig(outPath121, format='tif', bbox_inches='tight', transparent=True, dpi=300)
    plt.rcParams['svg.fonttype'] = 'none'
    outPath121 = outPath + '\\' + r'_XGB_ROC.svg'
    plt.savefig(outPath121, format='svg', bbox_inches='tight', transparent=True)
    plt.close()




