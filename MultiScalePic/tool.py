#_*_coding:utf-8_*_
__author__ = 'Alex_XT'
import os
import cv2
import imutils


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    print("(H:{},W:{})".format(image.shape[0], image.shape[1]))
    # compute the new dimensions of the image and resize it
    w = int(image.shape[1] / scale)
    image = imutils.resize(image, width=w)
    print("resize=(H:{},W:{})".format(image.shape[0], image.shape[1]))
    # if the resized image does not meet the supplied minimum
    # size, then stop constructing the pyramid
    if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
        print("Out of size!")
    else:
        yield image
def pyramid2(image, scale=1.5, minSize=(30, 30)):
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        print('(H:{},W:{})'.format(image.shape[0], image.shape[1]))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            print("Out of size!")
            break
        # yield the next image in the pyramid
        yield image

if __name__ == '__main__':
    image_path = r"E:\bamboo\IMG_20171201_131800.jpg"
    imageData = cv2.imread(image_path)
    pyramid(imageData,scale=8)
