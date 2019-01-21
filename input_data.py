#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:31:19 2017

@author: didizhang
"""

import tensorflow as tf
import numpy as np
import os

#%%
#返回图片列表和标签列表
train_dir = 'data/train/17_DEG/'        #训练文件
val_dir = 'data/test/15_DEG/'       #测试文件

def get_files(file_dir):        #返回存放文件的路径及对应标签
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    #构建图片类型列表
    BMP2 = []
    label_BMP2 = []
    BTR70 = []
    label_BTR70 = []
    T72 = []
    label_T72 = []
    S2S1=[]
    label_S2S1=[]
    BRDM_2=[]
    label_BRDM_2=[]
    D7=[]
    label_D7=[]
    SLICY=[]
    label_SLICY=[]
    T62=[]
    label_T62=[]
    ZIL131=[]
    label_ZIL131=[]
    ZSU_23_4=[]
    label_ZSU_23_4=[]    

    #将图片名进行拆分，并加入对应的类型列表
    for file in os.listdir(file_dir):
        name = file.split(sep='.') 
        if name[0]=='2S1':
            S2S1.append(file_dir + file)
            label_S2S1.append(0)
        if name[0]=='BRDM_2':
            BRDM_2.append(file_dir + file) 
            label_BRDM_2.append(1)
        if name[0]=='D7':
            D7.append(file_dir + file)
            label_D7.append(2)
        
        if name[0]=='T62':
            T62.append(file_dir + file) 
            label_T62.append(3)
        if name[0]=='ZIL131':
            ZIL131.append(file_dir + file)
            label_ZIL131.append(4)
        if name[0]=='ZSU_23_4':
            ZSU_23_4.append(file_dir + file)
            label_ZSU_23_4.append(5)
        if name[0]=='SLICY':
            SLICY.append(file_dir + file)
            label_SLICY.append(6)    
    '''
    namei=0
    for dicname in os.listdir(file_dir):
        for file in os.listdir(file_dir+dicname): 
            BMP2.append(file_dir+dicname +'/'+ file)
            label_BMP2.append(namei)
        namei=namei+1
  
    '''
    #print(BMP2)
    #print(label_BMP2)
    # print('There are %d BMP2\nThere are %d BTR70\nThere are %d T72' %(len(BMP2), len(BTR70),len(T72)))
    print('There are %d D7\nThere are %d BRDM_2\nThere are %d T62' % (len(D7), len(BRDM_2), len(T62)))      #输出类型数量
    # np.hstack:在水平方向上平铺
    image_list = np.hstack((BMP2, BTR70, T72,S2S1,BRDM_2,D7,SLICY,T62,ZIL131,ZSU_23_4))     #获得图片列表
    label_list = np.hstack((label_BMP2, label_BTR70, label_T72,label_S2S1,label_BRDM_2,label_D7,label_SLICY,label_T62,label_ZIL131,label_ZSU_23_4))     #获得标签分类列表

    temp = np.array([image_list, label_list])       #将图片列表和标签列表组合为一个矩阵
    #对二维数组的transpose操作就是对原数组的转置操作
    temp = temp.transpose()
    #现场修改序列，改变自身内容
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])       #图片列表
    label_list = list(temp[:, 1])       #标签列表
   
    label_list = [int(float(i)) for i in label_list]
    # print(image_list)
    #
    # print(label_list)
    
    return image_list, label_list


#%%
#生成相同大小的批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 1 ], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    #tf.cast(x, dtype, name=None) 将x的数据格式转化成dtype
    image = tf.cast(image, tf.string)       #将图片的格式转换成tf.string
    label = tf.cast(label, tf.int32)        #将标签的格式转换成tf.int32

    # make an input queue
    #建立input_queue:<class 'list'>: [<tf.Tensor 'input_producer/GatherV2:0' shape=() dtype=string>, <tf.Tensor 'input_producer/GatherV2_1:0' shape=() dtype=int32>]
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]      #label:Tensor("input_producer/GatherV2_1:0", shape=(), dtype=int32)
    #tf.read_file  读取图片
    image_contents = tf.read_file(input_queue[0])       #image_contents:Tensor("ReadFile:0", shape=(), dtype=string)
    #tf.image.decode_jpeg  将图片解码JPEG格式
    image = tf.image.decode_jpeg(image_contents, channels=1)        #图片解码   image：Tensor("random_crop:0", shape=(96, 96, 1), dtype=uint8)
    
    ######################################
    # data argumentation
    #tf.random_crop  图片随机裁剪
    image = tf.random_crop(image, [96, 96, 1])# randomly crop the image size to 96 x 96         #随机裁剪图片至大小96*96
    #tf.image.random_flip_left_right  图片随机左右翻转
    image = tf.image.random_flip_left_right(image)      #随机左右翻转图片
    #tf.image.random_brightness  图片随机调整亮度
    #image = tf.image.random_brightness(image, max_delta=63)        #随机调整图片亮度
    #tf.image.random_contrast  图片随机调整对比度
    image = tf.image.random_contrast(image,lower=0.2,upper=1.8)     #调整图片对比度

    ######################################

    #tf.image.resize_image_with_crop_or_pad  剪裁或填充处理，会根据原图像的尺寸和指定的目标图像的尺寸选择剪裁还是填充，如果原图像尺寸大于目标图像尺寸，则在中心位置剪裁，反之则用黑色像素填充
    #image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)        #改变图片尺寸至image_W,image_H
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    #如果您想测试生成的图像的批处理，您可能需要注释下面的行

    #tf.image.per_image_standardization  图片标准化
    image = tf.image.per_image_standardization(image)       #图片标准化

    #tf.train.batch  数据批量读取
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch
    #tf.train.shuffle_batch  通过随机打乱张量的顺序创建批次.
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)

    #reshape(tensor, shape, name=None)  tensor为被调整维度的张量。第2个参数为要调整为的形状。
    label_batch = tf.reshape(label_batch, [batch_size])     #调整标签维度
    image_batch = tf.cast(image_batch, tf.float32)      #将图片格式转换成float32
    
    return image_batch, label_batch


 
#%% TEST

# To test the generated batches of images
# When training the model, DO comment the following codes
"""
import matplotlib.pyplot as plt

BATCH_SIZE = 5
CAPACITY = 256
IMG_W = 128
IMG_H = 128

train_dir = '/Users/didizhang/Desktop/MSTAR/data/train/3_17_DEG/'

image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
 
with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop() and i<1:
            
            img, label = sess.run([image_batch, label_batch])
            
            # just test one batch
            for j in np.arange(BATCH_SIZE):
                print('label: %d' %label[j])
                plt.imshow(img[j,:,:,0],cmap='gray')
                plt.show()
            i+=1
            
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
"""