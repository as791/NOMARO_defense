# Copyright (C) 2021, Aryaman Sinha

# -*- coding: utf-8 -*-
"""make_blackbox_samples.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18o3yiYEc4U48SpR0sQ75i_aZtgH1LZtB
"""

# from google.colab import drive
# import os
# drive.mount('/content/drive/',force_remount=True)
# os.chdir('/content/drive/My Drive/data_generated/')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.applications.vgg16 import decode_predictions, preprocess_input
# from keras.applications.inception_v3 import  decode_predictions, preprocess_input
# from keras.applications.resnet import  decode_predictions, preprocess_input
from keras.preprocessing import image
import scipy.io as sio
from attack import square_attack_l2
import utils
import random
import os
import sys

class VGG16_:
  def __init__(self, session=None, use_log=False):
    self.image_size=224
    self.num_labels=1000
    self.num_channels=3

    input_= Input((224,224,3))
    input_1 = Lambda(lambda x: preprocess_input(x*255.0))(input_)
    if use_log:
      vgg_model = tf.keras.applications.VGG16(weights='imagenet',input_tensor=input_1)
    else:
      vgg_model = tf.keras.applications.VGG16(weights='imagenet',input_tensor=input_1,classifier_activation=None)
    self.model = Model(input_,vgg_model.output)

  def predict(self, data):
    return self.model(data)

class InceptionV3_:
  def __init__(self, session=None, use_log=False):
    self.image_size=299
    self.num_labels=1000
    self.num_channels=3

    input_= Input((299,299,3))
    input_1 = Lambda(lambda x: preprocess_input(x*255.0))(input_)
    if use_log:
      model = tf.keras.applications.InceptionV3(weights='imagenet',input_tensor=input_1)
    else:
      model = tf.keras.applications.InceptionV3(weights='imagenet',input_tensor=input_1,classifier_activation=None)
    self.model = Model(input_,model.output)

  def predict(self, data):
    return self.model(data)

class ResNet101_:
  def __init__(self, session=None, use_log=False):
    self.image_size=224
    self.num_labels=1000
    self.num_channels=3

    input_= Input((224,224,3))
    input_1 = Lambda(lambda x: preprocess_input(x*255.0))(input_)
    if use_log:
      model = tf.keras.applications.ResNet101(weights='imagenet',input_tensor=input_1)
    else:
      model = tf.keras.applications.ResNet101(weights='imagenet',input_tensor=input_1,classifier_activation=None)
    self.model = Model(input_,model.output)

  def predict(self, data):
    return self.model(data)

def readimg(ff):
  f = "./data/img/imgs/"+ff
  img = image.load_img(f,target_size=(299,299))
  img = image.img_to_array(img)
  if img.shape != (299, 299, 3):
    return None
  img = img/255
  return [img, int(ff.split(".")[0])-1]

class ImageNet:
  def __init__(self):
    np.random.seed(42)
    f = r"./data/img/imgs/"
    from multiprocessing import Pool
    pool = Pool(16)
    self.files = os.listdir(f)
    number = list(np.random.choice(range(len(self.files)),5000,replace=False))
    r = pool.map(readimg, [self.files[x] for x in number])
    r = [x for x in r if x != None]
    test_data, test_labels = zip(*r)
    self.test_data = np.array(test_data)
    self.test_labels = np.zeros((len(test_labels), 1000))
    self.test_labels[np.arange(len(test_labels)), test_labels] = 1

def generate_data(data, corr_pred, samples, targeted, start, vgg):
    inputs = []
    targets = []
    i=0
    cnt=0
    while(cnt<samples):
        if(corr_pred[start+i]):
            if targeted:
                if vgg:
                    seq = random.sample(range(0,1000), 10)
                else:
                    seq = range(data.test_labels.shape[1])

                for j in seq:
                    if (j == np.argmax(data.test_labels[start+i])) and (vgg == False):
                        continue
                    inputs.append(data.test_data[start+i])
                    targets.append(np.eye(data.test_labels.shape[1])[j])
            else:
                inputs.append(data.test_data[start+i])
                targets.append(data.test_labels[start+i])
            cnt+=1
        i+=1

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

#  def readimg(ff,key):
#   f = "./testing_database/vgg16/cw_l2/"+ff
#   arr = sio.loadmat(f)
#   img = arr[key]
#   return img

# def get_data():
#     np.random.seed(42)
#     f = "./testing_database/vgg16/cw_l2/"
#     test_data=[]
    
#     for x in os.listdir(f):
#       if 'orig' in x:
#         test_data.append(readimg(x,'orig'))
          
#     test_data = np.array(test_data)
#     return test_data, [x for x in os.listdir(f) if 'orig' in x]

with tf.Session() as sess:
  data = ImageNet()
  model = VGG16_(sess,use_log=False)
  # x_test,filenames = get_data()
  # pred = np.argmax(model.model.predict(x_test),-1)
  # test_labels = np.zeros((len(pred), 1000))
  # test_labels[np.arange(len(test_labels)), pred] = 1
  pred = np.argmax(model.model.predict(data.test_data),-1)
  corr_pred = (pred==np.argmax(data.test_labels,-1))
  acc = corr_pred.sum()/len(data.test_labels)
  print("Test Acc: {:.5f}".format(acc))
  square_attack = square_attack_l2
  p = 0.1 
  eps = 1275/255
  n_iter = 100000
  n_cls = 1000
  loss = 'margin_loss' 
  x_test, y_test = generate_data(data, corr_pred, samples=1000, targeted=False, start=0, vgg=True)
  logits_clean = model.model.predict(x_test)
  # corr_classified = logits_clean.argmax(1) == logits_clean.argmax(1)
  # y_target = pred
  # y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)
  corr_classified = logits_clean.argmax(1) == y_test.argmax(1)
  y_target = y_test.argmax(1)
  y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)

  timestart = time.time()
  adv=[]
  i=0
  cnt=0
  while(i<len(x_test)):
    n_queries, adv = square_attack(model, x_test[i:i+100], y_target_onehot[i:i+100], corr_classified[i:i+100], eps, n_iter,
      p, './metric_path/square_l2', False, loss)
    adv = np.array(adv)
    for j in range(len(adv)):
      adv_class = decode_predictions(model.model.predict(adv[j:j+1]))[0][0][1]
      orig_class = decode_predictions(model.model.predict(x_test[i+j:i+j+1]))[0][0][1]
      if adv_class!=orig_class:
        cnt+=1
        # name = './testing_database/vgg16/square_l2/cw_data/'+(filenames[i+j].split('_'))[0]+'_'+(filenames[i+j].split('_'))[1]+'_'+adv_class+'_adv.mat'
        # sio.savemat(name,{'adv':adv[j]})
        sio.savemat('./testing_database/vgg/square_l2/sample_'+str(i+j)+'_'+adv_class+'_adv.mat',{'adv':adv[j]})
    i=i+100
  timeend = time.time()

  print("Took",timeend-timestart,"seconds to run",len(x_test),"samples.")
  print("Attack success over correct predictions:", cnt/1000)

