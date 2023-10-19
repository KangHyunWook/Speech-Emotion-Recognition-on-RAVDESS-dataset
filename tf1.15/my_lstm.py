from config import get_config

from utils import *
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import sys
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import pickle
import tensorflow as tf
import numpy as np
import time
import math
import scipy.io as sio

n_time_step=10
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

rnn_suffix        =".mat_win_128_rnn_dataset.pkl"
label_suffix    =".mat_win_128_labels.pkl"


'''
load seed dataset of shape (128,9,9)
'''
from tensorflow.keras.utils import to_categorical
import scipy.io as scio
seed_dataset_path=r'/home/hyunwook_kang/SEED/ExtractedFeatures'

all_files = getPathList(seed_dataset_path)

file_map = getSubFileMap(all_files)
#todo:
true_labels=scio.loadmat(r'/home/hyunwook_kang/SEED/ExtractedFeatures/label.mat')['label'][0]

train_config = get_config()

# input parameter
n_input_ele = 310

input_channel_num = 1

n_labels = 3
# training parameter
lambda_loss_amount = 0.5
training_epochs = 500

# kernel parameter



kernel_height_3rd = 4
kernel_width_3rd = 4

kernel_height_4th = 1
kernel_width_4th = 1

kernel_stride = 1
conv_channel_num = 32

# algorithn parameter
learning_rate = 1e-4

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')


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

Y = tf.placeholder(tf.float32, shape=[None, n_labels], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')
rnn_in = tf.placeholder(tf.float32, shape=[None, n_time_step, n_input_ele], name='rnn_in')

rnn_in_flat = tf.reshape(rnn_in, [-1, 310])

rnn_fc_in = apply_fully_connect(rnn_in_flat, 310, n_fc_in)

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

y_ = apply_readout(lstm_fc_drop, lstm_fc_drop.shape[1], n_labels)
# print('y_', y_.shape)
# exit()
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
logdir = "my_tensorboard"
train_writer = tf.summary.FileWriter("log/"+logdir+"/train")
val_writer = tf.summary.FileWriter("log/"+logdir+"/val")
test_writer = tf.summary.FileWriter("log/"+logdir+"/test")

# set test batch number per epoch

subject_peak_acc ={}
for subject in range(1,16):

    subject=str(subject)
    train_x = []
    test_x = []

    train_y=[]
    test_y=[]
  
    for file in file_map[subject]:
        seed_data = scio.loadmat(file)
        for trial in range(1, 16):
            current_trial = seed_data[f'de_LDS{trial}']
            current_trial = current_trial.transpose(1, 0, 2)
            current_trial = current_trial.reshape(current_trial.shape[0], -1)

            if trial<10:
                train_x.extend(current_trial)
                train_y.extend([true_labels[trial-1]+1]*current_trial.shape[0])
            else:
                test_x.extend(current_trial)
                test_y.extend([true_labels[trial-1]+1]*current_trial.shape[0])

    train_x=np.asarray(train_x)
    test_x = np.asarray(test_x)
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    #todo: split train validation
   
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=7)
    batch_num_per_epoch = math.floor(train_x.shape[0]/train_config. batch_size)+ 1
    val_accuracy_batch_num = math.floor(val_x.shape[0]/train_config.batch_size)+1
    test_accuracy_batch_num = math.floor(test_x.shape[0]/train_config.batch_size)+ 1        
    best_val_loss=float('inf')
    with tf.Session(config=config) as session:
        print(f'================subject{subject}==================')
        val_writer.add_graph(session.graph)
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
                batch = train_x[start:(start+offset), :]
                
                rnn_batch=[]
                
                for j in range(batch.shape[0]):

                    rnn_batch.append(np.resize(batch[j], (10,310)))    

                rnn_batch = np.asarray(rnn_batch)                
                
                batch_y = train_y[start:(start+offset), :]       
                _ , c = session.run([optimizer, cost],
                                   feed_dict={rnn_in: rnn_batch, Y: batch_y,     
                                   keep_prob: 1 - dropout_prob,
                                              phase_train: True})
                
            
            for i in range(val_accuracy_batch_num):
                start = i* train_config.batch_size
                if (i+1)*train_config.batch_size>val_y.shape[0]:
                    offset = val_y.shape[0] % train_config.batch_size
                else:
                    offset = train_config.batch_size
            
                val_batch = val_x[start:(start + offset), :]
                
                val_cnn_batch=[]
                val_rnn_batch=[]
                for j in range(val_batch.shape[0]):
                    val_cnn_batch.append(np.resize(val_batch[j], (10,310,1)))
                    val_rnn_batch.append(np.resize(val_batch[j], (10,310)))    
                val_cnn_batch = np.asarray(val_cnn_batch)
                val_rnn_batch=np.asarray(val_rnn_batch)

                val_batch_y = val_y[start:(start + offset), :]

                tf_summary, val_a, val_c = session.run([merged,accuracy, cost],
                                               feed_dict={rnn_in: 
                                                val_rnn_batch,
                                                  Y: val_batch_y, keep_prob: 1.0, 
                                                            phase_train: False})
                val_writer.add_summary(tf_summary,val_count_accuracy)
                val_loss = np.append(val_loss, val_c)
                val_accuracy = np.append(val_accuracy, val_a)
                val_count_accuracy += 1
            print("Epoch: ", epoch + 1, " Val Cost: ",
                  np.mean(val_loss), "Val Accuracy: ", np.mean(val_accuracy))
            val_accuracy_save = np.append(val_accuracy_save, np.mean(val_accuracy))
            val_loss_save = np.append(val_loss_save, np.mean(val_loss))
     
            print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Test Cost: ",
                  np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy), "\n")
            if np.mean(val_loss) < best_val_loss:
                best_val_loss=np.mean(val_loss)
                for j in range(test_accuracy_batch_num):
                    start = j * train_config.batch_size
           
                    if (j+1)*train_config.batch_size>test_y.shape[0]:
                        offset = test_y.shape[0] % train_config.batch_size
                    else:
                        offset = train_config.batch_size
                    test_batch = test_x[start:(start + offset), :]

                    test_cnn_batch=[]
                    test_rnn_batch=[]
                    for j in range(test_batch.shape[0]):
                        test_cnn_batch.append(np.resize(test_batch[j], (10,310,1)))
                        test_rnn_batch.append(np.resize(test_batch[j], (10,310)))    
                    test_cnn_batch = np.asarray(test_cnn_batch)
                    test_rnn_batch = np.asarray(test_rnn_batch)

                    test_batch_y = test_y[start:(start + offset), :]
                    
                    tf_test_summary,test_a, test_c = session.run([merged,accuracy, cost],
                                                 feed_dict={rnn_in: 
                                                 test_rnn_batch,Y: test_batch_y,
                                                            keep_prob: 1.0, phase_train: False})
                    test_writer.add_summary(tf_test_summary,test_count_accuracy)
                    test_accuracy = np.append(test_accuracy, test_a)
                    test_loss = np.append(test_loss, test_c)
                    test_count_accuracy += 1
                mean_test_accuracy = np.mean(test_accuracy)
                subject_peak_acc[subject]=mean_test_accuracy
            print('peak accuracy:', subject_peak_acc[subject])


peak_accuracy_list=[]
print('=======peak accuracies========')
for i in range(1,16):
    print('subject{}: peak accuracy:'.format(i,subject_peak_acc[str(i)]))
    peak_accuracy_list.append(subject_peak_acc[str(i)])

print('l:', len(peak_accuracy_list))
f= open('subject_wise_acc_seed.csv', 'w')

for i in range(len(peak_accuracy_list)):
    f.write(str(np.round(100*peak_accuracy_list[i],2))+',')

f.close()
mean_acc=sum(peak_accuracy_list)/len(peak_accuracy_list)
print('mean accuracy of all subjects:', sum(peak_accuracy_list)/len(peak_accuracy_list))
print('std:', (sum((x-mean_acc)**2 for x in peak_accuracy_list) / (len(peak_accuracy_list)-1))**0.5)





#
