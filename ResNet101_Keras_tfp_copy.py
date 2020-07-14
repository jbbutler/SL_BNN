#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
"""
import numpy as np # linear algebra
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import ResNet50 ,ResNet50V2, ResNet101
import os
#from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow_probability as tfp
import tensorflow as tf
from glob import glob
from keras.utils.data_utils import get_file
from keras import backend as K
from tensorflow.keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.layers import Layer 
from keras import activations, initializers
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')
from keras.callbacks import EarlyStopping
import scipy.stats as stats
import math
import time
import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow_probability.python.layers import DenseFlipout
import h5py

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


time_init = time.time()

def load_data(path_X,path_Y,n_data):
    """Loads a dataset.
    # Arguments
        path: 
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    with np.load(path_X, allow_pickle=True) as f:
        X = f['X']
        print ("input X shape",X.shape)
        X = X#.transpose(1,0,2,3)
        st1_x = 0 
        ed1_x = 54000 
        st2_x = 54000 
        ed2_x = 60000 
        x_train = np.concatenate([X[st1_x:ed1_x],X[60000:114000]],axis=0)
        x_test = np.concatenate([X[st2_x:ed2_x],X[114000:120000]],axis=0)      

    with np.load(path_Y, allow_pickle=True) as f:
        Y,l_p = f['label'],f['Y']
        Y = Y#.transpose(1,0)
        l_p = l_p.transpose(1,0)
        st1_x = 0 
        ed1_x = 54000 
        st2_x = 54000 
        ed2_x = 60000 
        y_train = np.concatenate([Y[st1_x:ed1_x],Y[60000:114000]],axis=0)
        y_test = np.concatenate([Y[st2_x:ed2_x],Y[114000:120000]],axis=0)

        l_p_train = np.concatenate([l_p[st1_x:ed1_x],l_p[60000:114000]],axis=0)
        l_p_test = np.concatenate([l_p[st2_x:ed2_x],l_p[114000:120000]],axis=0)    


    return (x_train, y_train,l_p_train), (x_test, y_test,l_p_test)

################## DATA PREPARATION ###############################
###########################################

Data_lensed = True
# choose regression on lensed or unlensed data
data_train = True  ##### True used the traing data , False used the inference pipeline data
#label = '_ResNet101_DenseFlipout'
label = '_ResNet50_classification'
n_p = 3
train_reg_only_lensed = True
Train_ResNet101_Flipout = True
Train_BNN_ResNet50 = False
Load_PreTrain_BNN_ResNet50=False
Load_PreTrain_ResNet101_Flipout = False
Load_PreTrain_ResNet101_Dropout = False
Train_classification_resnet50 = False
Train_classify_model_save = False
PreTrain_classification_resnet50 = False
###########################################

###Jimmy's Addition to Access Data from Roomba Directories###
path_Jimmy = "/home/butlerj/projects/SL_Pipeline_KDD"
date = "6_9_20"



if (data_train):
   print ("loading training data")
   path_X =  path_Jimmy + "/KDD_Submission/Data/Array_HR.npz"
   #path_X = "/KDD_Submission/Data/Data_XYlabel_fimg_120.npz" #baseline
else:
   print ("loading Inference data")
   path_X = path_Jimmy + "/KDD_Submission/Denoising_and_Deblending/EDSR_MWCNN/experiment/Joint_model_inference/Array_SR_2.npz"


path_Y = path_Jimmy + "/KDD_Submission/Data/Data_Ylabel_fimg_120_new.npz"


#data_mat = np.load(path)['X']
data_mat_lp = np.load(path_Y)['Y'].T
n_data = 120000

n_data_lensed = 60000 

(x_train, y_train,l_p_train), (x_test, y_test,l_p_test) = load_data(path_X,path_Y,n_data)


l_p_train_radians = np.deg2rad(l_p_train[:,2])
l_p_test_radians = np.deg2rad(l_p_test[:,2])

e1_train =  ((1.0-l_p_train[:,1])/(1.0+l_p_train[:,1])) * np.cos(2*l_p_train_radians)
e2_train =  ((1.0-l_p_train[:,1])/(1.0+l_p_train[:,1])) * np.sin(2*l_p_train_radians)

e1_test =  ((1.0-l_p_test[:,1])/(1.0+l_p_test[:,1])) * np.cos(2*l_p_test_radians)
e2_test =  ((1.0-l_p_test[:,1])/(1.0+l_p_test[:,1])) * np.sin(2*l_p_test_radians)

l_p_train[:,1] = e1_train
l_p_train[:,2] = e2_train

l_p_test[:,1] = e1_test
l_p_test[:,2] = e2_test

l_p_train = l_p_train[:,0:n_p].reshape(-1,n_p)
l_p_test = l_p_test[:,0:n_p].reshape(-1,n_p)

max_lensed_inp = np.max(np.concatenate([l_p_train,l_p_test],axis=0),axis=0)
min_lensed_inp = np.min(np.concatenate([l_p_train,l_p_test],axis=0),axis=0)

print ("max_lensed_inp",max_lensed_inp)
print ("min_lensed_inp",min_lensed_inp)


if (Data_lensed):
    x_train_lensed = x_train[0:54000]
    l_p_train_lensed = l_p_train[0:54000]
    x_test_lensed = x_test[0:6000]
    l_p_test_lensed = l_p_test[0:6000]

else:
    x_train_lensed = x_train[54000:]
    l_p_train_lensed = l_p_train[54000:]
    x_test_lensed = x_test[6000:]
    l_p_test_lensed = l_p_test[6000:]

''' 
### best so far
### with Resnet100, except loss function looks weird

img_rows = 111
img_cols = 111
num_classes = 2
batch_size = 256
epochs = 200

noise = 0.1   #1
lr0 =  5e-2 #1e-1
dr = 0.01
stepsize = 4
'''


img_rows = 111
img_cols = 111
num_classes = 2
batch_size = 256
epochs = 1

noise = 0.1  
lr0 =  1e-3 
dr = 0.01
stepsize = 16




if K.image_data_format() == 'channels_first':
    print ("channels_first")
    
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)

    x_train_lensed = x_train_lensed.reshape(x_train_lensed.shape[0], 3, img_rows, img_cols)
    x_test_lensed = x_test_lensed.reshape(x_test_lensed.shape[0], 3, img_rows, img_cols)

else:
    print ("channels_last")
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

    x_train_lensed = x_train_lensed.reshape(x_train_lensed.shape[0], img_rows, img_cols, 3)
    x_test_lensed = x_test_lensed.reshape(x_test_lensed.shape[0], img_rows, img_cols, 3)

################## DATA: LENSED and UNLENSED  ############################

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalization of X
# if (data_extract == 'option1'):
#    x_train /= np.max(data_mat) #255 #Change this to max value over all channels?
#    x_test /= np.max(data_mat) #255  #Change this to max value

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

l_p_train_orig = l_p_train.astype('float32')
l_p_test_orig = l_p_test.astype('float32')
l_p_train = (l_p_train_orig-np.min(data_mat_lp,axis=0))/(np.max(data_mat_lp,axis=0) - np.min(data_mat_lp,axis=0)) 
l_p_test = (l_p_test_orig-np.min(data_mat_lp,axis=0))/(np.max(data_mat_lp,axis=0) - np.min(data_mat_lp,axis=0))
print('l_p_train shape:', l_p_train.shape)
num_features = l_p_train.shape[1]

################## DATA: ONLY LENSED  #####################################

x_train_lensed = x_train_lensed.astype('float32')
x_test_lensed = x_test_lensed.astype('float32')

print('x_train_lensed shape:', x_train_lensed.shape)
print(x_train_lensed.shape[0], '_lensed train samples')
print(x_test_lensed.shape[0], '_lensed test samples')

l_p_train_orig_lensed = l_p_train_lensed.astype('float32')
l_p_test_orig_lensed = l_p_test_lensed.astype('float32')
l_p_train_lensed = (l_p_train_orig_lensed-min_lensed_inp)/(max_lensed_inp - min_lensed_inp)
l_p_test_lensed = (l_p_test_orig_lensed-min_lensed_inp)/(max_lensed_inp - min_lensed_inp)
print('l_p_train_lensed shape:', l_p_train_lensed.shape)
num_features = l_p_train_lensed.shape[1]
y_test_orig = y_test.copy()

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr= 1e-3, decay_factor= dr, step_size= 500)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
optimizer = optimizers.Adam(lr=lr0)
#optimizer = optimizers.RMSprop(lr=0.0001)
lr_metric = get_lr_metric(optimizer)

if Train_classification_resnet50:
    from tensorflow.keras.applications import ResNet50

    batch_size = 512
    epochs = 150


    model = tf.keras.Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=None)) # change weights=?"imagenet"
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))#,
            #callbacks=[lr_sched])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])



    if (Train_classify_model_save):
        #model.save('./ResNet50_classification_limg_sim_Data_from_deblending.h5')
        model.save('./ResNet50_classification_hyperparam.h5')
        #model = load_model('ResNet50_SL.h5')
        print (model.summary())

    # Plot History
    fig=plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig('Metrics_'+label+'.png')

elif PreTrain_classification_resnet50:

    print ("loading classification model")
    model = load_model('./ResNet50_classification_hyperparam.h5')

    model_cl_pre = tf.keras.Sequential()
    for layer in model.layers[:]:
        model_cl_pre.add(layer)
    for layer in model_cl_pre.layers:
        layer.trainable = False

    print ("model_cl_pre-summary")
    print (model_cl_pre.summary()) 

    model_cl_pre.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', lr_metric])
    score = model_cl_pre.evaluate(x_test, y_test, verbose=0)     
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Calculate other classification metrics:
    yhat_probs = model_cl_pre.predict(x_test, verbose=0)
    yhat_classes = model_cl_pre.predict_classes(x_test, verbose=0)
    print ("np.shape(yhat_probs)",np.shape(yhat_probs),"np.shape(yhat_classes)",np.shape(yhat_classes))

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 1]
    #yhat_classes = yhat_classes[:]   



    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test_orig, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test_orig, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test_orig, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test_orig, yhat_classes)
    print('F1 score: %f' % f1)

    # kappa
    kappa = cohen_kappa_score(y_test_orig, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_test_orig, yhat_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(y_test_orig, yhat_classes)
    print(matrix)      


if Train_ResNet101_Flipout:

    epochs = 300
    batch_size = 256
    
    #from tensorflow.keras.applications import ResNet50 
    from tensorflow.keras.applications import ResNet50V2, ResNet101 
    model2 = tf.keras.Sequential()

    #model2.add(ResNet50V2(include_top=False, pooling='avg', weights=None))
    model2.add(ResNet101(include_top=False, pooling='max', weights=None))
    #model2.add(ResNet50(include_top=False, pooling='avg', weights=None))
    #model2.add(tfp.layers.DenseFlipout(512))
    #model2.add(tfp.layers.DenseFlipout(256))
    #model2.add(tfp.layers.DenseFlipout(64))
    #model2.add(tfp.layers.DenseFlipout(32))
    model2.add(tfp.layers.DenseFlipout(1024))
    #model2.add(tf.keras.layers.DenseFlipout(512, activation='linear'))
    #model2.add(tfp.layers.DenseFlipout(512))
    #model2.add(tf.keras.layers.DenseFlipout(128, activation='linear'))
    model2.add(tfp.layers.DenseFlipout(64))
    model2.add(tfp.layers.DenseFlipout(32))
    model2.add(tfp.layers.DenseFlipout(num_features))

    '''
    def neg_log_likelihood(y_obs, y_pred, sigma=noise):
        dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
        return K.sum(-dist.log_prob(y_obs))


    logits = model2(features)
    neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    kl = sum(model2.losses)
    elbo_loss = neg_log_likelihood + kl
    '''

    def elbo_loss(y_obs, y_pred, sigma=noise):
        dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
        neg_loglike =  K.sum(-dist.log_prob(y_obs))

        kl = K.sum(model2.losses)

        elbo_loss = neg_loglike + kl

        return elbo_loss
        
    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=0)

    #model2.compile(loss=neg_log_likelihood, optimizer=adam, metrics=['mse'])
    model2.compile(loss=elbo_loss, optimizer=adam, metrics=['mse','mae'])

    if (train_reg_only_lensed):
        Train_X  = x_train_lensed
        Train_lp = l_p_train_lensed
        Test_X   = x_test_lensed
        Test_lp = l_p_test_lensed
        Train_lp_orig = l_p_train_orig_lensed
        Test_lp_orig = l_p_test_orig_lensed
    else:
        Train_X  = x_train
        Train_lp = l_p_train
        Test_X   = x_test
        Test_lp = l_p_test
        Train_lp_orig = l_p_train_orig
        Test_lp_orig = l_p_test_orig



    history= model2.fit(Train_X, Train_lp, batch_size = batch_size, epochs = epochs, validation_data =(Test_X, Test_lp), verbose=1)

    file = h5py.File(path_Jimmy + '/models/' + date + '/' + str(time_init) + 'h5pyResNet101_regression_DenseFlipout.h5', 'w')
    weight = model2.get_weights()
    for i in range(len(weight)):
       file.create_dataset('weight' + str(i), data=weight[i])
    file.close()

    fig=plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig('Metrics_'+label+'.png')

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, epochs+1)
    fig = plt.figure(figsize=(10,10))
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.legend(['train_loss', 'val_loss'])
    fig.savefig(fname = path_Jimmy + '/model_metrics/' + date  + '/' +  str(time_init)+'val_train_history_'+label+'.png')

    #model2.save(path_Jimmy + '/models/' + date + '/' + str(time_init) + '100_BNN_ResNet101_regression_DenseFlipout.h5')

elif Train_BNN_ResNet50:

    epochs = 200
    batch_size = 256

    #conv_block of the bayesian ResNet50
    def conv_block(inputs,kernel_size, filters, strides=(2, 2)):
        filters1, filters2, filters3 = filters

        x = tfp.layers.Convolution2DFlipout(filters1, (1,1), strides = strides)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tfp.layers.Convolution2DFlipout(filters2, kernel_size, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tfp.layers.Convolution2DFlipout(filters3, (1, 1))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut = tfp.layers.Convolution2DFlipout(filters3, (1,1), strides=strides)(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def identity_block(inputs, kernel_size, filters):
        filters1, filters2, filters3 = filters
        x = tfp.layers.Convolution2DFlipout(filters1,(1,1))(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tfp.layers.Convolution2DFlipout(filters2, kernel_size, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tfp.layers.Convolution2DFlipout(filters3, (1,1))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.add([x, inputs])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def build_bayesian_resnet50(num_classes, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32')
        x = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
        x = tfp.layers.Convolution2DFlipout(64, (7, 7), strides=(2, 2), padding='VALID')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
        x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(x)
        x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256])
        x = identity_block(x, 3, [64, 64, 256])
        x = conv_block(x, 3, [128, 128, 512])
        x = identity_block(x, 3, [128, 128, 512])
        x = identity_block(x, 3, [128, 128, 512])
        x = identity_block(x, 3, [128, 128, 512])
        x = conv_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = conv_block(x, 3, [512, 512, 2048])
        x = identity_block(x, 3, [512, 512, 2048])
        x = identity_block(x, 3, [512, 512, 2048])
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tfp.layers.DenseFlipout(num_classes)(x)
        model = tf.keras.Model(inputs, x, name='resnet50')
        return model

    num_features = 3

    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=0)

    bayesian_resnet50 = build_bayesian_resnet50(num_features, (111,111,3))

    def elbo_loss(y_obs, y_pred, sigma=noise):
        dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
        neg_loglike =  K.sum(-dist.log_prob(y_obs))

        kl = K.sum(bayesian_resnet50.losses)

        elbo_loss = neg_loglike + kl

        return elbo_loss

    bayesian_resnet50.compile(loss=elbo_loss, optimizer=adam, metrics=['mse','mae'])

    if (train_reg_only_lensed):
        Train_X  = x_train_lensed
        Train_lp = l_p_train_lensed
        Test_X   = x_test_lensed
        Test_lp = l_p_test_lensed
        Train_lp_orig = l_p_train_orig_lensed
        Test_lp_orig = l_p_test_orig_lensed
    else:
        Train_X  = x_train
        Train_lp = l_p_train
        Test_X   = x_test
        Test_lp = l_p_test
        Train_lp_orig = l_p_train_orig
        Test_lp_orig = l_p_test_orig


    history= bayesian_resnet50.fit(Train_X, Train_lp, batch_size = batch_size, epochs = epochs, validation_data =(Test_X, Test_lp), verbose=1)

    file = h5py.File(path_Jimmy + '/models/' + date + '/' + str(time_init) + 'h5pyBNNResNet50_regression_DenseFlipout.h5', 'w')
    weight = bayesian_resnet50.get_weights()
    for i in range(len(weight)):
       file.create_dataset('weight' + str(i), data=weight[i])
    file.close()

    fig=plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig('Metrics_'+label+'.png')

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, epochs+1)
    fig = plt.figure(figsize=(10,10))
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.legend(['train_loss', 'val_loss'])
    fig.savefig(fname = path_Jimmy + '/model_metrics/' + date  + '/' +  str(time_init)+'val_train_history_'+label+'.png')

elif Load_PreTrain_BNN_ResNet50:

    if (train_reg_only_lensed):
        Train_X  = x_train_lensed
        Train_lp = l_p_train_lensed
        Test_X   = x_test_lensed
        Test_lp = l_p_test_lensed
        #np.save(path_Jimmy + '/model_metrics/lensed_labels.npy', Test_lp)
        Train_lp_orig = l_p_train_orig_lensed
        Test_lp_orig = l_p_test_orig_lensed
    else:
        Train_X  = x_train
        Train_lp = l_p_train
        Test_X   = x_test
        Test_lp = l_p_test
        Train_lp_orig = l_p_train_orig
        Test_lp_orig = l_p_test_orig

    #conv_block of the bayesian ResNet50
    def conv_block(inputs,kernel_size, filters, strides=(2, 2)):
        filters1, filters2, filters3 = filters

        x = tfp.layers.Convolution2DFlipout(filters1, (1,1), strides = strides)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tfp.layers.Convolution2DFlipout(filters2, kernel_size, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tfp.layers.Convolution2DFlipout(filters3, (1, 1))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut = tfp.layers.Convolution2DFlipout(filters3, (1,1), strides=strides)(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def identity_block(inputs, kernel_size, filters):
        filters1, filters2, filters3 = filters
        x = tfp.layers.Convolution2DFlipout(filters1,(1,1))(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tfp.layers.Convolution2DFlipout(filters2, kernel_size, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tfp.layers.Convolution2DFlipout(filters3, (1,1))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.add([x, inputs])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def build_bayesian_resnet50(num_classes, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32')
        x = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
        x = tfp.layers.Convolution2DFlipout(64, (7, 7), strides=(2, 2), padding='VALID')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
        x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(x)
        x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256])
        x = identity_block(x, 3, [64, 64, 256])
        x = conv_block(x, 3, [128, 128, 512])
        x = identity_block(x, 3, [128, 128, 512])
        x = identity_block(x, 3, [128, 128, 512])
        x = identity_block(x, 3, [128, 128, 512])
        x = conv_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = identity_block(x, 3, [256, 256, 1024])
        x = conv_block(x, 3, [512, 512, 2048])
        x = identity_block(x, 3, [512, 512, 2048])
        x = identity_block(x, 3, [512, 512, 2048])
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tfp.layers.DenseFlipout(num_classes)(x)
        model = tf.keras.Model(inputs, x, name='resnet50')
        return model

    num_features = 3

    bayesian_resnet50 = build_bayesian_resnet50(num_features, (111,111,3))

    file = h5py.File(path_Jimmy + '/models/' + date + '/1594001215.681334h5pyBNNResNet50_regression_DenseFlipout.h5', 'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight' + str(i)][:])
    bayesian_resnet50.set_weights(weight)

    print(bayesian_resnet50.summary())

    lp_test_pred = np.zeros((len(Test_lp),50,3))

    for i in tqdm.tqdm(range(50)):
        preds = bayesian_resnet50.predict(Test_X)
        lp_test_pred[:,i,:] = preds

    np.save(path_Jimmy + '/model_metrics/' + date  + '/' + str(time_init) +  'BNN_ResNet50_DenseFlipoutPredMatrix.npy', lp_test_pred)

elif Load_PreTrain_ResNet101_Flipout:

    if (train_reg_only_lensed):
        Train_X  = x_train_lensed
        Train_lp = l_p_train_lensed
        Test_X   = x_test_lensed
        Test_lp = l_p_test_lensed
        #np.save(path_Jimmy + '/model_metrics/lensed_labels.npy', Test_lp)
        Train_lp_orig = l_p_train_orig_lensed
        Test_lp_orig = l_p_test_orig_lensed
    else:
        Train_X  = x_train
        Train_lp = l_p_train
        Test_X   = x_test
        Test_lp = l_p_test
        Train_lp_orig = l_p_train_orig
        Test_lp_orig = l_p_test_orig


    from tensorflow.keras.applications import ResNet50V2, ResNet101
    model2 = tf.keras.Sequential()

    #model2.add(ResNet50V2(include_top=False, pooling='avg', weights=None))
    model2.add(ResNet101(include_top=False, pooling='max', weights=None))
    #model2.add(ResNet50(include_top=False, pooling='avg', weights=None))
    #model2.add(tfp.layers.DenseFlipout(512))
    #model2.add(tfp.layers.DenseFlipout(256))
    #model2.add(tfp.layers.DenseFlipout(64))
    #model2.add(tfp.layers.DenseFlipout(32))
    model2.add(tfp.layers.DenseFlipout(1024))
   # model2.add(tfp.layers.DenseFlipout(512))
    #model2.add(tfp.layers.DenseFlipout(256))
    #model2.add(tf.keras.layers.DenseFlipout(128, activation='linear'))
    #model2.add(tfp.layers.DenseFlipout(64))
    #model2.add(tfp.layers.DenseFlipout(32))
    model2.add(tfp.layers.DenseFlipout(num_features))


    file = h5py.File(path_Jimmy + '/models/' + date + '/1591683040.7190077h5pyResNet101_regression_DenseFlipout.h5', 'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight' + str(i)][:])
    model2.set_weights(weight)
    
    print(model2.summary())


    #Uncomment this line if you plan to do inference directly after training
    #model2 = load_model(path_Jimmy + '/models/' + date + '/' + str(time_init) + '100_BNN_ResNet101_regression_DenseFlipout.h5')

    #Use this if you are running the code and wish to save a loaded model directly
    #model2 = load_model(path_Jimmy + '/models/' + date + '/1589923682.6864126100_BNN_ResNet101_regression_DenseFlipout.h5')

    lp_test_pred = np.zeros((len(Test_lp),50,3))

    for i in tqdm.tqdm(range(50)):
        preds = model2.predict(Test_X)
        lp_test_pred[:,i,:] = preds


    ## Save the prediction matrix
    np.save(path_Jimmy + '/model_metrics/' + date  + '/' + str(time_init) +  'ResNet101_DenseFlipoutPredMatrix.npy', lp_test_pred)

    #lp_test_pred_mean = np.mean(lp_test_pred, axis=1)
    #lp_test_pred_std = np.std(lp_test_pred, axis=1)

    #lp_test_pred_mean = np.array(lp_test_pred_mean)
    #lp_test_pred_std = np.array(lp_test_pred_std)
    
    #fig, ax = plt.subplots(nrows=2, ncols=n_p,figsize=(20, 20))

    #ax[0, 0].vlines(x=Test_lp[:, 0], ymin=lp_test_pred_mean[:, 0]-lp_test_pred_std[:, 0], ymax=lp_test_pred_mean[:, 0]+lp_test_pred_std[:, 0], alpha=0.05)
    #ax[0, 1].vlines(x=Test_lp[:, 1], ymin=lp_test_pred_mean[:, 1]-lp_test_pred_std[:, 1], ymax=lp_test_pred_mean[:, 1]+lp_test_pred_std[:, 1], alpha=0.05)
    #ax[0, 2].vlines(x=Test_lp[:, 2], ymin=lp_test_pred_mean[:, 2]-lp_test_pred_std[:, 2], ymax=lp_test_pred_mean[:, 2]+lp_test_pred_std[:, 2], alpha=0.05)

    #ax[0, 0].set_xlabel('true')
    #ax[0, 0].set_ylabel('pred')
    #ax[0, 1].set_xlabel('true')
    #ax[0, 1].set_ylabel('pred')
    #ax[0, 2].set_xlabel('true')
    #ax[0, 2].set_ylabel('pred')

    #ax[0, 0].plot( [0, 1], [0, 1], 'r')
    #ax[0, 1].plot( [0, 1], [0, 1], 'r')
    #ax[0, 2].plot( [0, 1], [0, 1], 'r')

    #ax[0, 0].set_ylim([-2.5, 2.5])
    #ax[0, 1].set_ylim([-2.5, 2.5])
    #ax[0, 2].set_ylim([-2.5, 2.5])


    #fig.savefig(path_Jimmy + '/model_metrics/' + date + '/' + str(time_init)+'_BNN_uq_plot.png')


    #print ("lp_test_pred_mean.shape",lp_test_pred_mean.shape)

    #RMSE_test_norm = np.sqrt(np.mean((lp_test_pred_mean-Test_lp)**2))
    #MAE_test_norm  = (np.mean(np.abs(lp_test_pred_mean-Test_lp)))

    #print (RMSE_test_norm,MAE_test_norm)

    #lp_test_trnsf = lp_test_pred_mean*(max_lensed_inp - min_lensed_inp) + (min_lensed_inp)

    #RMSE_test_orig = np.sqrt(np.mean((lp_test_trnsf-Test_lp_orig)**2))
    #MAE_test_orig  = (np.mean(np.abs(lp_test_trnsf-Test_lp_orig)))

    #print (RMSE_test_orig,MAE_test_orig)

