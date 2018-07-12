'''
Created on 2017-4-8

@author: XuTing
'''
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def read_and_decode(filename):
    filename_queue=tf.train.string_input_producer([filename])#生成一个queue队列
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)#返回文件名和文件
    features=tf.parse_single_example(serialized_example, features={
       'label':tf.FixedLenFeature([],tf.int64),
       'img_raw':tf.FixedLenFeature([],tf.string)
       })#将image数据和label取出来
    img=tf.decode_raw(features['img_raw'], tf.uint8)
    img=tf.reshape(img, [200,200,3])#reshape为128*128的3通道图片
#     img = tf.cast(img, tf.float32) *(1./255)-0.5#在流中抛出img张量，训练时要标准化！！！！
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    return img, label

batchsize=5000#查找bug标号-1
# tfrecords_file=("E:\KaggleDogsVsCats\TFData\dog_vs_cat_train.tfrecords")
tfrecords_file ='E://Parking186//Train_TF//train.tfrecords'
image,label=read_and_decode(tfrecords_file)
# save_dir='E:\CarVsMicrovanVsMiniCarVsMPVVsPickupVsSUVVsTruck\ReshapeImage'
save_dir = 'E://Parking186//ImgReshape//'
save_path=os.path.join(save_dir,"DogVsCat_")

sess=tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
coord=tf.train.Coordinator()
threads= tf.train.start_queue_runners(sess=sess,coord=coord)
for i in range(batchsize):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        img.save(save_path+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片

print("Save to patch:",save_path+"   Successful")
coord.request_stop()
coord.join(threads)


    
    
    
    
    
    
    
    