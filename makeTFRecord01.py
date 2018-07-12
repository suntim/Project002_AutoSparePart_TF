'''
Created on 2017-4-8

@author: XuTing
'''
import os
import tensorflow as tf
from PIL import Image
import numpy as np

# cwd = 'E://CarVsMicrovanVsMiniCarVsMPVVsPickupVsSUVVsTruck//ZToTruck//'
cwd = 'E://Parking186//Train_InitData//'
# save_dir='E://CarVsMicrovanVsMiniCarVsMPVVsPickupVsSUVVsTruck//TestTF//'
save_dir='E://Parking186//Train_TF//'
filename=os.path.join(save_dir,"train.tfrecords")
classes = {'0','1'}

# cwd='E:\KaggleDogsVsCats\TFRecordInit\\'
# save_dir='E:\KaggleDogsVsCats\TFData'
# filename=os.path.join(save_dir,"dog_vs_cat_train.tfrecords")
# classes={'cat','dog'}#必须为两类的文件名字  cat=0 dog=1，但是这字典会随机变换！！！
writer=tf.python_io.TFRecordWriter(filename)#要生成的文件名
num = 0
count = 0
images = []
labels = [] 
temp = []
TFimages = []
TFlabels = []
TFtemp=[]


for index,name in enumerate(classes):
    class_path=cwd+name+'\\'#读取类路径
    for img_name in os.listdir(class_path):
        img_path=class_path+img_name#获取每一个图片地址
        img=Image.open(img_path)
        images.append(img_path)
        labels.append(index)
#         print(min(img.size))#if (num >= 0 and num <= 5) or (num >= 10 and num <= 15):    
        if (min(img.size)<200) or (os.path.getsize(img_path) >=479000):
            print('~~error%s，Ruined Images!: %s'%(count,img_path))
        else:
            num+=1
            img=img.resize((200,200))#裁剪大小,,,,,,要不要进行标准化处理？
            img_raw=img.tobytes()#将图片转化为二进制格式
            example=tf.train.Example(features=tf.train.Features(feature={
                 "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                 'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))#example对象对label和image数据进行封装
            TFimages.append(img_path)
            TFlabels.append(index)
            writer.write(example.SerializeToString())#序列化为字符串
        count += 1
temp = np.array([images, labels])
temp = temp.transpose()#column?
TFtemp = np.array([TFimages, TFlabels])
TFtemp = TFtemp.transpose()#column?
print("Save to patch:",save_dir+"   Successful")
print("总可用样本：",num)
writer.close()

    



