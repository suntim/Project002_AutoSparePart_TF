#_*_coding:utf-8_*_

import os
import sys
import re
import cv2
import  numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lib import tools


image_dir = r"E:\CarVsMicrovanVsMiniCarVsMPVVsPickupVsSUVVsTruck\Train_Init\Truck\1111"
save_dir = r"E:\CarVsMicrovanVsMiniCarVsMPVVsPickupVsSUVVsTruck\Train_Init\Truck\save"

for i,fileName in enumerate(os.listdir(image_dir)):
    #print("{}-----{}".format(i,content))
    if re.match(".*.jpg$",fileName):
        img_path = os.path.join(image_dir,fileName)
        print(img_path)
        image=cv2.imread(img_path)
        tools.check_dir(save_dir)
        index = 1
        for resized_Img in tools.pyramid2(image,scale=2,minSize=(30,30)):
            # print(type(resized_Img))
            # cv2.imshow(fileName,resized_Img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(save_dir,"resz_{}".format(str(index).zfill(3))+fileName),resized_Img)
            index += 1

print("Done!")
