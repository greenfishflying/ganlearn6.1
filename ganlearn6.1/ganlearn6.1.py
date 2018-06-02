# -*- coding: utf-8 -*-#
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.legacy.layers import MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconv2D, UpSampling2D
from BatchNorm_GAN import BatchNormGAN
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
#import matplotlib.pyplot as plt
import pickle, random, sys, keras
from keras.models import Model
from IPython import display

sys.path.append("../common")
from keras.utils import np_utils
from tqdm import tqdm

K.set_image_dim_ordering('th')

#准备数据
img_rows,img_cols=28,28
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255
# 看看我们所有数据的个数和长相
print(np.min(X_train),np.max(X_train))
#print(X_train[0])
print("xtrainshape:",X_train.shape)
print("y_train:",y_train)
print("len(Y_train):",len(y_train))

# 做个指示器，告诉算法，现在这个net（要么是dis要么是gen），能不能被继续train
def make_trainable(net,val):
	net.trainable=val
	for l in net.layers:
		l.trainable=val
		
shp = X_train.shape[1:]
yshp=y_train[0]
print(shp)
print(yshp)
dropout_rate =0.25
# 设置gen和dis的opt
# 大家可以尝试各种组合
opt = Adam(lr=1e-3)
dopt = Adam(lr=1e-4)

nch=200

g_input=Input(shape=[100])

# 倒过来的CNN第一层（也就是普通CNN那个flatten那一层）
H=Dense(nch*14*14,init='glorot_normal')(g_input)
#H=BatchNorm_GAN()(H)
H = BatchNormGAN()(H)
H=Activation('relu')(H)
H=Reshape([nch,14,14])(H)
H=UpSampling2D(size=(2,2))(H)
# CNN滤镜
H=Convolution2D(int(nch/2),3,3,border_mode='same',init='glorot_uniform')(H)
H = BatchNormGAN()(H)
H=Activation('relu')(H)
H=Convolution2D(int(nch/4),3,3,border_mode='same',init='glorot_uniform')(H)
H = BatchNormGAN()(H)
H=Activation('relu')(H)
H=Convolution2D(1,1,1,border_mode='same',init='glorot_uniform')(H)
g_V =Activation("sigmoid")(H)
generator=Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
#generator.summary()

#创建dis
d_input=Input(shape=shp)
H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
# 滤镜
H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
# flatten之后，接MLP
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
# 出一个结果，『是』或者『不是』
d_V = Dense(2,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
#discriminator.summary()

make_trainable(discriminator,False)
gan_input=Input(shape=[100])
H=generator(gan_input)
gan_V=discriminator(H)
GAN=Model(gan_input,gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()

ntrain=10000
trainidx=random.sample(range(0,X_train.shape[0]),ntrain)
XT = X_train[trainidx,:,:,:]

noise_gen = np.random.uniform(0,1,size=[XT.shape[0],100])
#generated=generator.predict(noise_gen)
generated_images = generator.predict(noise_gen)
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y=np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1
make_trainable(discriminator,True)
discrimnator.fit(X,y,nb_epoch=1, batch_size=32)
y_hat = discriminator.predict(X)
y_hat_idx=np.argmax(y_hat,axis=1)
print(len(y_hat))
y_idx=np.argmax(y,axis=1)
diff=y_idx-y_hat_idx
n_tot=y.shape[0]
n_rig=(diff==0).sum()
acc = n_rig*100.0/n_tot
print("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))