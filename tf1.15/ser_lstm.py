from sklearn.model_selection import StratifiedShuffleSplit

from config import get_config

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
import sys
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

import pickle5 as pickle
import tensorflow as tf
import numpy as np
import time
import math
import scipy.io as sio
import os
import librosa



def preprocess(au_mfcc_path):
    data = []
    labels = []
    with open(au_mfcc_path, 'rb') as f:
        au_mfcc = pickle.load(f)

    print(len(au_mfcc))

    for key in au_mfcc:
        emotion = key.split('-')[2]
        emotion = int(emotion)-1
        labels.append(emotion)
        data.append(au_mfcc[key])

    data=np.array(data)
    labels = np.array(labels)
    labels = labels.reshape(labels.shape+(1,))

    data = np.hstack((data, labels))
    fdata = shuffle(data)
    data = fdata[:, :-1]
    labels = fdata[:, -1].astype(int)

    return data, labels

data_path = 'au_mfcc.pkl'

data, labels=preprocess(data_path)

new_labels= np.zeros((labels.shape[0], np.unique(labels.astype(int)).size))
for i in range(len(labels)):
    new_labels[i, labels[i]]=1

labels=new_labels

test_data=data[-181:-1]
test_labels=labels[-181:-1]
data=data[:-180]
labels=labels[:-180]

train_data=data[:1020]
train_labels=labels[:1020]


print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)

train_X = train_data
test_X = test_data 


n_time_step=1
input_height = 10
input_width = 310

n_lstm_layers = 2

# lstm full connected parameter
n_hidden_state = 32
print("\nsize of hidden state", n_hidden_state)
n_fc_out = 1024
n_fc_in = 1024

dropout_prob = 0.5

norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = True



from tensorflow.keras.utils import to_categorical
import scipy.io as scio


train_config = get_config()

# input parameter
n_input_ele = 294

input_channel_num = 1

n_classes = 8
Y = tf.placeholder(tf.float32, shape=[None, n_classes], name='true_labels')
# training parameter
lambda_loss_amount = 0.5
training_epochs = 500

kernel_stride = 1

# algorithn parameter
learning_rate = 1e-4

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1],
                          strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

def apply_fully_connect(x, x_size, fc_size):
    
    initial = tf.truncated_normal([x_size, fc_size], stddev=0.1)
    fc_weight = tf.Variable(initial)
    fc_bias = bias_variable([fc_size])
    
    return tf.nn.elu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size):
    readout_weight = weight_variable([x_size, readout_size])
    readout_bias = bias_variable([readout_size])
    # print('r2:', readout_bias.shape)
    # exit()
    return tf.add(tf.matmul(x, readout_weight), readout_bias)



phase_train = tf.placeholder(tf.bool, name='phase_train')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')
rnn_in = tf.placeholder(tf.float32, shape=[None, n_time_step, n_input_ele], name='rnn_in')

rnn_in_flat = tf.reshape(rnn_in, [-1, 294])

initial = tf.truncated_normal([294, n_fc_in], stddev=0.1)
fc_weight = tf.Variable(initial)
fc_bias = bias_variable([n_fc_in])

rnn_fc_in = tf.nn.elu(tf.add(tf.matmul(rnn_in, fc_weight), fc_bias))

lstm_in = tf.reshape(rnn_fc_in, [-1, n_time_step, n_fc_in])

cells = []
for _ in range(n_lstm_layers):
    with tf.name_scope("LSTM_"+str(n_lstm_layers)):
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_state, forget_bias=1.0, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)

lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

output, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in,dtype=tf.float32, time_major=False)

output = tf.unstack(tf.transpose(output, [1, 0, 2]), name='lstm_out')

rnn_output = output[-1]

shape_rnn_out = rnn_output.get_shape().as_list()

lstm_fc_out = apply_fully_connect(rnn_output, shape_rnn_out[1], n_fc_out)

lstm_fc_drop = tf.nn.dropout(lstm_fc_out, keep_prob)

y_ = apply_readout(lstm_fc_drop, lstm_fc_drop.shape[1], n_classes)

y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
y_posi = tf.nn.softmax(y_, name="y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name='loss')
    tf.summary.scalar('cost_with_L2',cost)
else:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name='loss')
    tf.summary.scalar('cost',cost)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
tf.summary.scalar('accuracy',accuracy)

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

merged = tf.summary.merge_all()



train_y=train_labels
test_y = test_labels

print(train_X.shape, test_X.shape)
print(train_y.shape, test_y.shape)



#todo: split train validation

batch_num_per_epoch = math.floor(train_X.shape[0]/train_config. batch_size)+ 1
test_accuracy_batch_num = math.floor(test_X.shape[0]/train_config.batch_size)+ 1        

best_val_acc=-1

with tf.Session(config=config) as session:

    val_count_accuracy = 0
    test_count_accuracy = 0

    session.run(tf.global_variables_initializer())
    val_accuracy_save = np.zeros(shape=[0], dtype=float)
    test_accuracy_save = np.zeros(shape=[0], dtype=float)
    test_loss_save = np.zeros(shape=[0], dtype=float)
    val_loss_save = np.zeros(shape=[0], dtype=float)
    for epoch in range(training_epochs):
        val_accuracy = np.zeros(shape=[0], dtype=float)
        test_accuracy = np.zeros(shape=[0], dtype=float)
        test_loss = np.zeros(shape=[0], dtype=float)
        val_loss = np.zeros(shape=[0], dtype=float)
        
        for b in range(batch_num_per_epoch):
            start = b * train_config.batch_size
            if (b+1)*train_config.batch_size>train_y.shape[0]:
                offset = train_y.shape[0] % train_config.batch_size
            else:
                offset = train_config.batch_size
            train_batch = train_X[start:(start+offset), :]
            train_batch=np.expand_dims(train_batch,axis=1)         
            batch_y = train_y[start:(start+offset), :]    
          
            _ , c = session.run([optimizer, cost],
                               feed_dict={rnn_in: train_batch, Y: batch_y,     
                               keep_prob: 1 - dropout_prob,
                                          phase_train: True})
            
        
        for i in range(test_accuracy_batch_num):
            start = i* train_config.batch_size
            if (i+1)*train_config.batch_size>test_y.shape[0]:
                offset = test_y.shape[0] % train_config.batch_size
            else:
                offset = train_config.batch_size
        
            test_batch = test_X[start:(start + offset), :]
            test_batch=np.expand_dims(test_batch,axis=1)   
            test_batch_y = test_y[start:(start + offset), :]

            tf_summary, val_a, val_c = session.run([merged,accuracy, cost],
                                           feed_dict={rnn_in: 
                                            test_batch,
                                              Y: test_batch_y, keep_prob: 1.0, 
                                                        phase_train: False})
            val_loss = np.append(val_loss, val_c)
            val_accuracy = np.append(val_accuracy, val_a)
            val_count_accuracy += 1
        print("Epoch: ", epoch + 1, " Val Cost: ",
              np.mean(val_loss), "Val Accuracy: ", np.mean(val_accuracy))
        val_accuracy_save = np.append(val_accuracy_save, np.mean(val_accuracy))
        val_loss_save = np.append(val_loss_save, np.mean(val_loss))
 
        print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Test Cost: ",
              np.mean(val_loss), "Test Accuracy: ", np.mean(val_accuracy), "\n")
        if np.mean(val_accuracy) > best_val_acc:
            best_val_acc=np.mean(val_accuracy)
            peak_acc=best_val_acc
            print('peak accuracy:', np.round(100*peak_acc,2))











#
