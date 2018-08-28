# 180827
# Trung-Hieu Tran @ IPVS

import keras
import json
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json

from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import regularizers

import numpy as np
from time import strftime, gmtime
import matplotlib.pyplot as plt
import tensorflow as tf

## LOAD DATA
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print(" Number of training data: %d , number of testing data %d"%(len(x_train),len(x_test)))

# hyperparameters
n_epoch = 70
batch_size = 256

# encoded code dimensional
encoding_dim = 64

# input place holder
input_img = Input(shape=(784,)) # 28 x 28
# encoded = Dense(encoding_dim,activation='relu')(input_img)
encoded = Dense(encoding_dim,
                activity_regularizer=regularizers.l1(10e-8),
                activation=None)(input_img)

decoded = Dense(784,activation='sigmoid')(encoded)
# define a model for end-to-end training
autoencoder = Model(input_img,decoded)

# define a model to get intermediate result
encoder = Model(input_img,encoded)
encoded_input = Input(shape=(encoding_dim,))
print(" Layers of autoencoder ", autoencoder.layers)
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input,decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# setting up backend, callbacks
config = tf.ConfigProto(device_count={'GPU':4,'CPU':4})
sess = tf.Session(config=config)
keras.backend.set_session(sess)
# tensorboard logs
str_time = strftime("%y%m%d_%H%M%S",gmtime())
tensorboard = TensorBoard(log_dir="logs/{}".format(str_time))

# feeding data for tranining
autoencoder.fit(x_train,x_train,
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tensorboard])
# evaluate
scores = autoencoder.evaluate(x_test,x_test)
print("%s: %.2f%%"%(autoencoder.metrics_names[1], scores[1]*100))

# reialize model to JSON
model_json = autoencoder.to_json()
parsed = json.loads(model_json)
with open("./logs/model_{}.json".format(str_time),'w') as json_file:
    json_file.write(json.dumps(parsed, indent=4, sort_keys=True))
print("Saved model to disk")
# serialize weights to HDF5
autoencoder.save_weights("./logs/model_{}.h5".format(str_time))
print("Saved weight to disk")
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# Visualization
n_img = 10
encoded_imgs = encoder.predict(x_test[0:n_img,:])
decoded_imgs = decoder.predict(encoded_imgs)
fig = plt.figure(figsize=(20,4))
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

fig.savefig("./logs/output_{}.png".format(str_time))

