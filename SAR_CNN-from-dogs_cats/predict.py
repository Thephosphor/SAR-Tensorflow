#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Thephosphor
"""
import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
image_size=64
num_channels=3
images = []
path = r'ZIL131.HB14932.JPG'           #选择需要识别的图片路径
image = cv2.imread(path)            #读取图片
# 将图像调整到我们想要的大小并进行预处理
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)           #改变图片大小
images.append(image)
images = np.array(images, dtype=np.uint8)           #转换图片为矩阵
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)             #矩阵对应元素位置相乘
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)         #矩阵变形
## Let us restore the saved model           #恢复模型
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./ZIL-ZSU-model/ZIL-ZSU.ckpt-550.meta')         #重建网络图,此步骤只创建图
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, './ZIL-ZSU-model/ZIL-ZSU.ckpt-550')         #加载使用restore方法保存的权重
# Accessing the default graph which we have restored
graph = tf.get_default_graph()          #访问已恢复的默认图
# Now, let's get hold of the op that we can be processed to get the output.         #处理op，得到输出
# In the original network y_pred is the tensor that is the prediction of the network            #原始网络中，y_pred是网络的预测张量
y_pred = graph.get_tensor_by_name("y_pred:0")           #引用保存的操作和占位符变量
## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")          #引用保存的操作和占位符变量
y_true = graph.get_tensor_by_name("y_true:0")           #引用保存的操作和占位符变量
y_test_images = np.zeros((1, 3))            #构建初始化矩阵
### Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
res_label = ['2S1','BRDM_2','D7','SLICY','T62','ZIL131','ZSU_23_4']          #图片的内容分类标签
print("This is {} with possibility {}".format(res_label[result.argmax()],max(result[0])))           #输出图片识别结果类型及可能的概率