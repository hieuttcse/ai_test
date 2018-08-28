# 180824
# Trung-Hieu Tran @ IPVS

import keras
from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist
from keras.callbacks import TensorBoard

import numpy as np
from time import strftime, gmtime
import matplotlib.pyplot as plt
import tensorflow as tf


##LOAD DATA
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
print(" Train shape: ", x_train.shape, " Test shape: ", x_test.shape)
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print(" Train shape: ", x_train.shape, " Test shape: ", x_test.shape)

# epoch
n_epoch = 50
# batchsize
batch_size = 256

# encoded code dimensional
encoding_dim = 32

# input place holder
input_img = Input(shape=(784,)) #mnist with input size 28 x 28

# simple fully mapped net for encoded code
encoded = Dense(encoding_dim,activation='relu')(input_img)
# encoded = Dense(encoding_dim)(input_img)

# another simple net for decoding
decoded = Dense(784,activation='sigmoid')(encoded)
# decoded = Dense(784)(encoded)


# define model for end-to-end traning
autoencoder = Model(input_img, decoded)

# define models for analyzing intermediate result
encoder = Model(input_img,encoded)
encoded_code = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_code, decoder_layer(encoded_code))

autoencoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy')
# autoencoder.compile(optimizer='sgd',
#                     loss='mean_squared_error')

# setup backend, callbacks
config = tf.ConfigProto( device_count={'GPU':1, 'CPU':4})
sess = tf.Session(config=config)
keras.backend.set_session(sess)
# tensorboard logs
str_time = strftime("%y%m%d_%H%M%S",gmtime())
tensorboard = TensorBoard(log_dir="logs/{}".format(str_time))

autoencoder.fit(x_train,x_train,
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test,x_test),
                callbacks=[tensorboard])

# VISUALIZATIOn
n_img=10
encoded_imgs = encoder.predict(x_test[0:n_img,:])
decoded_imgs = decoder.predict(encoded_imgs)
plt.figure(figsize=(20,4))
for i in range(n_img):
    # display original
    ax = plt.subplot(2,n_img,i+1)
    plt.imshow(x_test[i].reshape(28,28))
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

print("Mean of encoded_imgs: ",encoded_imgs.mean())


