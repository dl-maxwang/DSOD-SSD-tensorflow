"""
"""

import os
import gc
import xml.etree.ElementTree as etxml
import random
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import ssd300
import time
import cv2

from dataset.Dataset import Dataset

'''
SSD检测
'''

batch_size = 1
class_size=22
graph_config = 'dsod'
dataset = Dataset(batch_size=batch_size, img_preprocess_fn=lambda x: x - whitened_RGB_mean)

def testing():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssd_model = ssd300.SSD300(sess, False, class_size=class_size, graph_config=graph_config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index'):
            saver.restore(sess, './session_params/session.ckpt')
            image, actual, file_list = next(dataset.next_test())
            pred_class, pred_class_val, pred_location = ssd_model.run(image, None, file_list)
            print('file_list:' + str(file_list))
            #
            for index, act in zip(range(len(image)), actual):
                for a in act:
                    print('【img-' + str(index) + ' actual】:' + str(a))
                print('pred_class:' + str(pred_class[index]))
                print('pred_class_val:' + str(pred_class_val[index]))
                print('pred_location:' + str(pred_location[index]))

        else:
            print('No Data Exists!')
        sess.close()


'''
SSD训练
'''


def training():

    running_count = 0

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssd_model = ssd300.SSD300(sess, True, class_size=class_size, graph_config=graph_config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index'):
            print('\nStart Restore')
            saver.restore(sess, './session_params/session.ckpt')
            print('\nEnd Restore')

        print('\nStart Training')
        min_loss_location = 100000.
        min_loss_class = 100000.

        while ((min_loss_location + min_loss_class) > 0.001 and running_count < 100000):
            running_count += 1

            train_data, actual_data, _ = next(dataset.next_train())
            print(_)
            if len(train_data) > 0:
                loss_all, loss_class, loss_location, pred_class, pred_location = ssd_model.run(train_data, actual_data, _)
                l = np.sum(loss_location)
                c = np.sum(loss_class)
                if min_loss_location > l:
                    min_loss_location = l
                if min_loss_class > c:
                    min_loss_class = c

                print('Running:【' + str(running_count) + '】|Loss All:【' + str(
                    min_loss_location + min_loss_class) + '|' + str(loss_all) + '】|Location:【' + str(
                    np.sum(loss_location)) + '】|Class:【' + str(np.sum(loss_class)) + '】|pred_class:【' + str(
                    np.sum(pred_class)) + '|' + str(np.amax(pred_class)) + '|' + str(
                    np.min(pred_class)) + '】|pred_location:【' + str(np.sum(pred_location)) + '|' + str(
                    np.amax(pred_location)) + '|' + str(np.min(pred_location)) + '】')

                # 定期保存ckpt
                if running_count % 100 == 0:
                    saver.save(sess, './session_params/session.ckpt')
                    print('session.ckpt has been saved.')
                    gc.collect()
            else:
                print('No Data Exists!')
                break

        saver.save(sess, './session_params/session.ckpt')
        sess.close()
        gc.collect()

    print('End Training')


'''
获取voc2007训练图片数据
train_data：训练批次图像，格式[None,width,height,3]
actual_data：图像标注数据，格式[None,[None,center_x,center_y,width,height,lable]]
'''
# 图像白化，格式:[R,G,B]
whitened_RGB_mean = [123.68, 116.78, 103.9]

'''
主程序入口
'''
if __name__ == '__main__':
    print('\nStart Running')
    # 检测
    # testing()
    # 训练
    training()
    print('\nEnd Running')
