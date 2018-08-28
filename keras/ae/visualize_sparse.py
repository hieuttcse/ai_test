# 180828
# Trung-Hieu Tran @ IPVS

import keras
from keras.models import Model, Sequential, model_from_json

from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# LOAD Data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print(" Number of training data: %d , number of testing data %d"%(len(x_train),len(x_test)))

model_fname = "./logs/model_180828_151110"
# load json and create model
with open("{}.json".format(model_fname),'r') as json_file:
    model_json = json_file.read()
autoencoder = model_from_json(model_json)
# load weights into new model
autoencoder.load_weights("{}.h5".format(model_fname))
print("Loaded model from disk")

n_img = 10
start = 11
decoded_imgs = autoencoder.predict(x_test[start: start+n_img,:])
fig = plt.figure(figsize=(20,4))
for i in range(n_img):
    # display original
    ax = plt.subplot(2,n_img,i+1)
    plt.imshow(x_test[start+i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display decoded
    ax = plt.subplot(2,n_img, i+1+n_img)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

