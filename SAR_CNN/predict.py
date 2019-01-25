from scipy import misc
import tensorflow as tf
import numpy as np
import os, glob
import pandas as pd
# 输入图像，下面代码将目录下所有图片输入
size_image = 96
num_channels = 1
images = []
classes = ['BTR_60', '2S1', 'BRDM_2', 'D7', 'SLICY', 'T62', 'ZIL131', 'ZSU_23_4']
target_class = []

### 测试目录内所有图片
# path = 'train'
# # im = misc.imread(fl)
# for fields in classes:
#     index = classes.index(fields)
#     print('Now going to read {} files (Index: {})'.format(fields, index))
#     path = os.path.abspath('train')
#
#     for root, dirs, files in os.walk(path, topdown=True):
#         for direct in dirs:
#             if direct == fields:
#                 path = os.path.join(root, direct, '*.jpg')
#
#                 files1 = glob.glob(path)
#                 for fl in files1:
#                     im = misc.imread(fl)
#                     im = misc.imresize(im, [size_image, size_image])
#                     im = np.atleast_3d(im)
#                     images.append(im)
#                     target_class.append(fields)
#                     # print(fl)

### 测试单个图片
path = r'E:\CQM_GitFiles\SAR demo TensorFlow\data\TEST\15_DEG\T62.HB15201.JPG'
im = misc.imread(path)
im = misc.imresize(im, [size_image, size_image])
im = np.atleast_3d(im)
images.append(im)
images = np.array(images)
target_class = np.array(target_class)
# print("Number of files in Testing-set:\t\t{}".format(images.shape[0]))
x_batch = images
sess = tf.Session()
# 导入图
saver = tf.train.import_meta_graph('sar-model/model-199.meta')
saver.restore(sess, tf.train.latest_checkpoint('sar-model/'))
# 恢复图
graph = tf.get_default_graph()
# 预测输出
y_pred = graph.get_tensor_by_name("y_pred:0")
## 图像输入
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 8))
###计算结果   这里可 修改为单图
feed_dict_testing = {x: x_batch, y_true: y_test_images}
# 属于各分类的分值
result = sess.run(y_pred, feed_dict=feed_dict_testing)
# 概率化处理
tgt_class = sess.run(tf.argmax(result, 1), feed_dict=feed_dict_testing)
print("This is {} with possibility {}".format(classes[result.argmax()],max(result[0])))
# df2 = pd.DataFrame(target_class)
# df2.to_csv('TargetClassNames.csv')
# df = pd.DataFrame(result)
# df.to_csv('Results.csv')
# df1 = pd.DataFrame(tgt_class)
# df1.to_csv('ClassesIndices.csv')