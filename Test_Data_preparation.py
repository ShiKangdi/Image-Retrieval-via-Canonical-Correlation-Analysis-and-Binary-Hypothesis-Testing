import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import os
import cv2
np.set_printoptions(precision=8)

Max_ResultM = np.load('Max_ResultM.npy')
Ave_ResultM = np.load('Ave_ResultM.npy')
Std_ResultM = np.load('Std_ResultM.npy')

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
        keys = ['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 'conv2_2_W', 'conv2_2_b', 'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', 'conv4_1_W', 'conv4_1_b', 'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b', 'conv5_1_W', 'conv5_1_b', 'conv5_2_W', 'conv5_2_b', 'conv5_3_W', 'conv5_3_b']
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))

            
if __name__ == '__main__':
    print('Evaluation_1')
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32)
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)  
    NameList = []
    AllFolder = 'E:/New/Oxford/oxbuild_images2' #
    QueryFolder = 'E:/New/Oxford/crop_img'
    All_Query = ['1_crop_all_souls_000013.jpg','2_crop_all_souls_000026.jpg',
                 '3_crop_oxford_002985.jpg', '4_crop_all_souls_000051.jpg',
                 '5_crop_oxford_003410.jpg','6_crop_ashmolean_000058.jpg',
                 '7_crop_ashmolean_000000.jpg','8_crop_ashmolean_000269.jpg',
                 '9_crop_ashmolean_000007.jpg','10_crop_ashmolean_000305.jpg',
                 '11_crop_balliol_000051.jpg','12_crop_balliol_000187.jpg',
                 '13_crop_balliol_000167.jpg','14_crop_balliol_000194.jpg',
                 '15_crop_oxford_001753.jpg','16_crop_bodleian_000107.jpg',
                 '17_crop_oxford_002416.jpg','18_crop_bodleian_000108.jpg',
                 '19_crop_bodleian_000407.jpg','20_crop_bodleian_000163.jpg',
                 '21_crop_christ_church_000179.jpg','22_crop_oxford_002734.jpg',
                 '23_crop_christ_church_000999.jpg','24_crop_christ_church_001020.jpg',
                 '25_crop_oxford_002562.jpg','26_cornmarket_000047.jpg',
                 '27_cornmarket_000105.jpg','28_cornmarket_000019.jpg',
                 '29_oxford_000545.jpg','30_cornmarket_000131.jpg',
                 '31_hertford_000015.jpg','32_oxford_001752.jpg',
                 '33_oxford_000317.jpg','34_hertford_000027.jpg',
                 '35_hertford_000063.jpg','36_keble_000245.jpg',
                 '37_keble_000214.jpg','38_keble_000227.jpg',
                 '39_keble_000028.jpg','40_keble_000055.jpg',
                 '41_magdalen_000078.jpg','42_oxford_003335.jpg',
                 '43_magdalen_000058.jpg','44_oxford_001115.jpg','45_magdalen_000560.jpg',
                 '46_pitt_rivers_000033.jpg','47_pitt_rivers_000119.jpg',
                 '48_pitt_rivers_000153.jpg','49_pitt_rivers_000087.jpg',
                 '50_pitt_rivers_000058.jpg','51_radcliffe_camera_000519.jpg',
                 '52_oxford_002904.jpg','53_radcliffe_camera_000523.jpg',
                 '54_radcliffe_camera_000095.jpg','55_bodleian_000132.jpg']
    
    ChannelM = np.arange(512)  
    max_ResultA_P5 = []
    max_ResultB_P5 = []        
    ave_ResultA_P5 = []
    ave_ResultB_P5 = []
    std_ResultA_P5 = []
    std_ResultB_P5 = []
    
    for kk in range(len(All_Query)):
        QFile = All_Query[kk]    
        Qimg = imread(QueryFolder +'/' + QFile , mode='RGB')
        QP5 = sess.run(vgg.pool5, feed_dict={vgg.imgs: [Qimg]})[0]
        QP5 = np.float64(QP5)
        QP5 = np.reshape(QP5,(np.shape(QP5)[0]*np.shape(QP5)[1],512))

        max_DPA = QP5[np.argmax(QP5[:,ChannelM],axis=0),ChannelM] # max pooling
        ave_DPA = np.mean(QP5,axis=0) # average pooling
        std_DPA = np.std(QP5,axis=0,ddof=1) # std pooling
        
        max_DPA = max_DPA - Max_ResultM
        max_S_A = np.sqrt(np.sum(max_DPA**2))
        max_norm_A = max_DPA/max_S_A
        max_norm_A = np.nan_to_num(max_norm_A)
        max_ResultA_P5.append(max_norm_A) 
        
        ave_DPA = ave_DPA - Ave_ResultM
        ave_S_A = np.sqrt(np.sum(ave_DPA**2))
        ave_norm_A = ave_DPA/ave_S_A
        ave_norm_A = np.nan_to_num(ave_norm_A)
        ave_ResultA_P5.append(ave_norm_A) 
        
        std_DPA = std_DPA - Std_ResultM   
        std_S_A = np.sqrt(np.sum(std_DPA**2))
        std_norm_A = std_DPA/std_S_A
        std_norm_A = np.nan_to_num(std_norm_A)
        std_ResultA_P5.append(std_norm_A) 
            
        NameList.append(QFile)
  
    for i in range(len(os.listdir(AllFolder))):
        File = os.listdir(AllFolder)[i]    
        img = imread(AllFolder +'/' + File , mode='RGB')
        if max(np.shape(img))>1024:
            img = imresize(img, ((1024/max(np.shape(img)))*np.shape(img)[0], (1024/max(np.shape(img)))*np.shape(img)[1]))
        P5 = sess.run(vgg.pool5, feed_dict={vgg.imgs: [img]})[0]
        P5 = np.float64(P5)
        P5 = np.reshape(P5,(np.shape(P5)[0]*np.shape(P5)[1],512))
        
        max_DPB = P5[np.argmax(P5[:,ChannelM],axis=0),ChannelM] # max pooling
        ave_DPB = np.mean(P5,axis=0) # average pooling
        std_DPB = np.std(P5,axis=0,ddof=1) # std pooling
        
        max_DPB = max_DPB - Max_ResultM
        max_S_B = np.sqrt(np.sum(max_DPB**2))
        max_norm_B = max_DPB/max_S_B
        max_norm_B = np.nan_to_num(max_norm_B)
        max_ResultB_P5.append(max_norm_B) 
        
        ave_DPB = ave_DPB - Ave_ResultM
        ave_S_B = np.sqrt(np.sum(ave_DPB**2))
        ave_norm_B = ave_DPB/ave_S_B
        ave_norm_B = np.nan_to_num(ave_norm_B)
        ave_ResultB_P5.append(ave_norm_B) 
        
        std_DPB = std_DPB - Std_ResultM              
        std_S_B = np.sqrt(np.sum(std_DPB**2))
        std_norm_B = std_DPB/std_S_B
        std_norm_B = np.nan_to_num(std_norm_B)
        std_ResultB_P5.append(std_norm_B) 

        NameList.append(File)

    tf.reset_default_graph()   
    
    Max_Rshape_p5 = np.concatenate((max_ResultA_P5, max_ResultB_P5), axis=0)
    Ave_Rshape_p5 = np.concatenate((ave_ResultA_P5, ave_ResultB_P5), axis=0)
    Std_Rshape_p5 = np.concatenate((std_ResultA_P5, std_ResultB_P5), axis=0)
    
    np.save('Eva_Max_Rshape_p5.npy',Max_Rshape_p5)
    np.save('Eva_Ave_Rshape_p5.npy',Ave_Rshape_p5)
    np.save('Eva_Std_Rshape_p5.npy',Std_Rshape_p5)
    np.save('Eva_NameList.npy',NameList)
