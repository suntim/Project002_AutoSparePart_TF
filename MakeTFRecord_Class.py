# -*- coding: UTF-8 -*-MakeTFRecord.py
#!/usr/bin/env python
import tensorflow as tf
import os,sys
# from matplotlib import pyplot as plt
from scipy.misc import imread


def get_files_list(filename):
    file = open(filename, 'r')
    images_filename_list = [line.strip() for line in file]
    return images_filename_list

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def create_tfrecord_dataset(images_dir,filename_list, labelKey, writer):
    # create training tfrecord
    for i, file_name in enumerate(filename_list):
        image_name,label_name = file_name.split(',')
        print("{}--->{}--->{}".format(image_name,label_name,labelKey[label_name]))
        image_np = None
        img_path = os.path.join(images_dir, image_name.strip())
        try:
            image_np = imread(img_path)
            print("img_path = ",img_path)
        except FileNotFoundError:
            # read from Pascal VOC path
            print(os.path.join(images_dir, image_name.strip() + ".jpg"))

        image_h = image_np.shape[0]
        image_w = image_np.shape[1]

        img_raw = image_np.tostring()
        # img_raw = image_np.tobytes()  # 将图片转化为二进制格式
        # example = tf.train.Example(features=tf.train.Features(feature={
        #     'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labelKey[label_name]])),
        #     'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        # }))  # example对象对label和image数据进行封装
        # writer.write(example.SerializeToString())  # 序列化为字符串

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_h),
            'width': _int64_feature(image_w),
            'label': _int64_feature(int(labelKey[label_name])),
            'image_raw': _bytes_feature(img_raw)}))

        writer.write(example.SerializeToString())

    print("End of TfRecord. Total of image written:", i+1)
    writer.close()



if __name__ == '__main__':
    # define base paths for pascal the original VOC dataset training images
    images_dir = r"/home/alex/Downloads/JiLi/Dataset/dat/train"#图片地址
    train_label_txt_path = r"/home/alex/Downloads/JiLi/Dataset/dat/training_labels.txt"#训练的图片标签列表txt
    labelKeyTxt_path = r"/home/alex/Downloads/JiLi/Dataset/dat/unique_labels.txt"#符号标签转数字对应的标签txt
    SaveTF_DIR = r"/home/alex/Downloads/JiLi/Dataset/dat"
    TRAIN_FILE = 'train.tfrecords'
    labelKey = {}
    with open(labelKeyTxt_path,'r') as f:
        for line in f:
            content = line.strip().split(" ")
            labelKey[content[0]] = content[1]
    print(labelKey)

    filename_list = get_files_list(train_label_txt_path)
    print("Total number of training images:", len(filename_list))
    # print(filename_list)

    # shuffle array and separate 10% to validation
    # np.random.shuffle(filename_list)
    train_writer = tf.python_io.TFRecordWriter(os.path.join(SaveTF_DIR, TRAIN_FILE))

    # create training dataset
    create_tfrecord_dataset(images_dir,filename_list, labelKey, train_writer)
