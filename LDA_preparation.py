import tensorflow as tf
import numpy as np
import os
import numpy.matlib
import cv2
np.set_printoptions(precision=8)

Max_Rshape = np.load('Max_Rshape.npy') # Features from Max pooling after Centralization and Normalization
Ave_Rshape = np.load('Ave_Rshape.npy') # Features from Ave pooling after Centralization and Normalization
Std_Rshape = np.load('Std_Rshape.npy') # Features from Std pooling after Centralization and Normalization

Max_ResultM = np.mean(Max_Rshape,axis = 0) 
Ave_ResultM = np.mean(Ave_Rshape,axis = 0)
Std_ResultM = np.mean(Std_Rshape,axis = 0)

Max_mean = np.load('Max_ResultM.npy') # Mean of features from Max pooling before Centralization and Normalization 
Max_mean = np.reshape(Max_mean,(512,1))
Ave_mean = np.load('Ave_ResultM.npy') # Mean of features from Ave pooling before Centralization and Normalization 
Ave_mean = np.reshape(Ave_mean,(512,1)) 
Std_mean = np.load('Std_ResultM.npy') # Mean of features from Std pooling before Centralization and Normalization 
Std_mean = np.reshape(Std_mean,(512,1)) 

class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')


    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
#        keys = sorted(weights.keys())
        keys = ['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 'conv2_2_W', 'conv2_2_b', 'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', 'conv4_1_W', 'conv4_1_b', 'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b', 'conv5_1_W', 'conv5_1_b', 'conv5_2_W', 'conv5_2_b', 'conv5_3_W', 'conv5_3_b']
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))

            
if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32)
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)  
    print('1stCCA_1')
    FILE = 'D:/LM120K/All'
    A = os.listdir(FILE)
    
    ResultA_P5 = []
    ResultB_P5 = []
    
    NameList = []                                 
    max_Result_P5 = []        
    ave_Result_P5 = []
    std_Result_P5 = []
    max_Sb = np.zeros((512,512))
    max_Sw = np.zeros((512,512))
    ave_Sb = np.zeros((512,512))
    ave_Sw = np.zeros((512,512))
    std_Sb = np.zeros((512,512))
    std_Sw = np.zeros((512,512))
    Channel = range(512)
    l = 0
    for i in range(len(A)):
        max_Result_P5 = []        
        ave_Result_P5 = []
        std_Result_P5 = []
        Folder = A[i]
        File_len = len(os.listdir(FILE+'/'+os.listdir(FILE)[i]))
        for j in range(File_len):
            print([i,j,l])
            l = l+1
            
            File = os.listdir(FILE+'/'+ Folder)[j]
            img = cv2.imread(FILE +'/'+ Folder +'/'+ File)
            img = img[:,:,::-1]
            P5 = sess.run(vgg.pool5, feed_dict={vgg.imgs: [img]})[0]
            P5 = np.float64(P5)
            P5 = np.reshape(P5,(np.shape(P5)[0]*np.shape(P5)[1],512))
            
            max_DP = P5[np.argmax(P5[:,Channel],axis=0),Channel] # max pooling
            max_DP = np.reshape(max_DP,(512,1))
            ave_DP = np.mean(P5,axis=0) # average pooling
            ave_DP = np.reshape(ave_DP,(512,1))
            std_DP = np.std(P5,axis=0,ddof=1) # std pooling
            std_DP = np.reshape(std_DP,(512,1))
            
            Max_Rshape_p5_1= np.reshape((max_DP-Max_mean),(512,1))
            max_S_B = np.sqrt(np.sum(Max_Rshape_p5_1**2))
            max_norm_B = Max_Rshape_p5_1/max_S_B
            max_norm_B = np.nan_to_num(max_norm_B)
            max_Result_P5.append(max_norm_B)      
            
            
            Ave_Rshape_p5_1=np.reshape((ave_DP- Ave_mean),(512,1))
            ave_S_B = np.sqrt(np.sum(Ave_Rshape_p5_1**2))
            ave_norm_B = Ave_Rshape_p5_1/ave_S_B
            ave_norm_B = np.nan_to_num(ave_norm_B)
            ave_Result_P5.append(ave_norm_B)
                       
            
            Std_Rshape_p5_1=np.reshape((std_DP- Std_mean),(512,1))
            std_S_B = np.sqrt(np.sum(Std_Rshape_p5_1**2))
            std_norm_B = Std_Rshape_p5_1/std_S_B
            std_norm_B = np.nan_to_num(std_norm_B)
            std_Result_P5.append(std_norm_B)
            
            tf.reset_default_graph()
            
        max_mean =  np.mean(max_Result_P5,axis=0)
        ave_mean =  np.mean(ave_Result_P5,axis=0)
        std_mean =  np.mean(std_Result_P5,axis=0)
        
        max_dc = max_mean-Max_ResultM
        max_dc = np.reshape(max_dc,(1,512))
        max_Sb_ind = File_len*np.dot(np.transpose(max_dc),max_dc)
        max_Sb = max_Sb + max_Sb_ind
        
        ave_dc = ave_mean-Ave_ResultM
        ave_dc = np.reshape(ave_dc,(1,512))
        ave_Sb_ind = File_len*np.dot(np.transpose(ave_dc),ave_dc)
        ave_Sb = ave_Sb + ave_Sb_ind
        
        std_dc = std_mean-Std_ResultM
        std_dc = np.reshape(std_dc,(1,512))
        std_Sb_ind = File_len*np.dot(np.transpose(std_dc),std_dc)
        std_Sb = std_Sb + std_Sb_ind
        
        dc_max_Result_P5 = max_Result_P5 - max_mean
        dc_ave_Result_P5 = ave_Result_P5 - ave_mean
        dc_std_Result_P5 = std_Result_P5 - std_mean
        
        max_Sw_class = np.zeros((512,512))
        ave_Sw_class = np.zeros((512,512))
        std_Sw_class = np.zeros((512,512))
        for k in range(File_len):
            new_dc_max_Result_P5 = np.reshape(dc_max_Result_P5[k],(1,512))
            max_Sw_class_ind = np.dot(np.transpose(new_dc_max_Result_P5),new_dc_max_Result_P5)
            max_Sw_class = max_Sw_class + max_Sw_class_ind
            
            new_dc_ave_Result_P5 = np.reshape(dc_ave_Result_P5[k],(1,512))
            ave_Sw_class_ind = np.dot(np.transpose(new_dc_ave_Result_P5),new_dc_ave_Result_P5)
            ave_Sw_class = ave_Sw_class + ave_Sw_class_ind
            
            new_dc_std_Result_P5 = np.reshape(dc_std_Result_P5[k],(1,512))
            std_Sw_class_ind = np.dot(np.transpose(new_dc_std_Result_P5),new_dc_std_Result_P5)
            std_Sw_class = std_Sw_class + std_Sw_class_ind
            
        max_Sw = max_Sw + max_Sw_class
        ave_Sw = ave_Sw + ave_Sw_class
        std_Sw = std_Sw + std_Sw_class
                
    np.save('max_Sw.npy',max_Sw)
    np.save('ave_Sw.npy',ave_Sw)
    np.save('std_Sw.npy',std_Sw)    
    np.save('max_Sb.npy',max_Sb)
    np.save('ave_Sb.npy',ave_Sb)
    np.save('std_Sb.npy',std_Sb)
            
