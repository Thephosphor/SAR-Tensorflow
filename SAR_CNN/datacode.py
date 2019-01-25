from PIL import Image
from scipy import misc
import os
import glob
from sklearn.utils import shuffle
import numpy as np
from os.path import join, getsize


def load_train(path, size_image, classes):
    images = []
    labels = []
    img_names = []
    cls = []
    print('Going to read training images')
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.abspath('train') 
        
        for root, dirs, files in os.walk(path, topdown=True):
            for direct in dirs:
                if direct == fields:
                    path=os.path.join(root, direct, '*.jpg')
                    
                    files1 = glob.glob(path) 
                    for fl in files1:
                        im = misc.imread(fl)
                        im = misc.imresize(im, [size_image, size_image])
                        im = np.atleast_3d(im)
                        images.append(im)                              
                        lbl = np.zeros(len(classes))
                        lbl[index] = 1.0
                        labels.append(lbl)
                        flbase = os.path.basename(fl) 
                        img_names.append(flbase)
                        cls.append(fields) 
    
    #有些重复了，但random多次，打乱顺序
    temp = np.array([images, labels,img_names,cls])
    temp = temp.transpose()
    np.random.shuffle(temp)
    np.random.shuffle(temp)
    images = list(temp[:, 0]) 
    labels = list(temp[:, 1]) 
    img_names=list(temp[:,2])
    cls = list(temp[:, 3])
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` 数据."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      print(self._num_examples)
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(path, size_image, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(path, size_image, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  
 
  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets