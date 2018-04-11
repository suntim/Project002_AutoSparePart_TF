'''
Created on 2017年9月9日

@author: admin
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from skimage import transform
W = 200
H = 150
def recognize(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read()) #rb
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            input_x = sess.graph.get_tensor_by_name("input:0")
            print (input_x)
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            print (out_softmax)
            out_label = sess.graph.get_tensor_by_name("output:0")
            print (out_label)

            img = np.array(Image.open(jpg_path).convert('L')) 
            img = transform.resize(img, (H, W, 3))
            
            plt.figure("fig1")
            plt.imshow(img)
            img = img * (1.0 /255)
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(img, [-1, H, W, 3])})

            print ("img_out_softmax:",img_out_softmax)
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print ("prediction_labels:",prediction_labels)
            
            plt.show()

recognize("D:/AutoSparePart/ToFinall_Data/0/crop_or_pad235.jpg", "./output/autosparepart.pb")
