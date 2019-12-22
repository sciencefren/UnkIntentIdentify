import os
import numpy as np

import tensorflow as tf

from dataset import Dataset
from model import MultiLabelsClassificationModel as Model


def train(dataset):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = Model(embedding_dim=768,
                        max_sentence_len=512,
                        num_classes=d.num_classes,
                        num_clusters=d.num_clusters)
            # check variables
            for v in tf.trainable_variables():
                #if 'intent' in v.name:
                print(v.name)

            pwc_ckpt_path = '../../../../UnkIntentIdentifyModel/multilabels_cnn_cla_ckpt/'
            checkpoint = tf.train.latest_checkpoint(pwc_ckpt_path)
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)

            sess.run(tf.global_variables_initializer())
            if checkpoint:
                saver.restore(sess, checkpoint)

            print_info_steps = 2000
            svae_ckpt_steps = 10000
            while True:
                (batch_x, batch_y) = dataset.gen_next_labeled_batch(batch_size=64).__next__()
                _, loss, step = sess.run([model.train_op,
                    model.loss,
                    model.global_step],
                                             feed_dict={model.input_x: batch_x,
                                                        model.input_y: batch_y,
                                                        model.drop_keep_pro: 0.6})
                if not (step+1)%print_info_steps:
                    print("[PWC] step: {}, loss: {:.4}".format(step,
                        loss))
                if loss < 0.07:
                    saver.save(sess, pwc_ckpt_path+'model', step)
                    break
                if not (step+1)%svae_ckpt_steps:
                    saver.save(sess, pwc_ckpt_path+'model', step)
                

if __name__ == '__main__':
    d = Dataset(num_clusters=20)
    d.read_train_data(dataset_fp='../../../data/stackoverflow/train.csv',
            texts_vec_fp='../../../data/stackoverflow/train_vec.txt',
            label2y_fp='../../../data/stackoverflow/label2y_dct.pkl')
    d.split_data_by_label()

    train(dataset=d)
