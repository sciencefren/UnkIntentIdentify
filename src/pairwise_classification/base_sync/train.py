import os
import numpy as np

import tensorflow as tf

from dataset import Dataset
from model import PairwiseClassificationModel


def train(dataset):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = PairwiseClassificationModel(embedding_dim=768,
                                                num_clusters=d.num_clusters)
            # check variables
            for v in tf.trainable_variables():
                #if 'intent' in v.name:
                print(v.name)

            pwc_ckpt_path = os.path.join('../../../UnkIntentIdentifyModel/pwc_mt_ckpt', 'model.ckpt')
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, pwc_ckpt_path)

            lambda_stable_steps = 0
            print_info_steps = 200
            while True:
                (batch_x, batch_y) = dataset.gen_next_labeled_batch(batch_size=256).__next__()
                (batch_x_unlabeled, _) = dataset.gen_next_unlabeled_batch(batch_size=256).__next__()
                _, loss, step = sess.run([model.train_op, model.loss, model.global_step],
                                             feed_dict={model.input_x: batch_x,
                                                        model.input_x_unlabeled: batch_x_unlabeled,
                                                        model.input_y: batch_y,
                                                        model.drop_keep_pro: 0.6})
                bound_delta = sess.run(model.bound_delta)
                if bound_delta <= 0:
                    lambda_stable_steps += 1
                if not (lambda_stable_steps+1)%print_info_steps:
                    print("[PWC] step: {}, loss: {:.4}, bound delta: {:.4}".format(step, loss, bound_delta))
                if lambda_stable_steps >= 400000:
                    saver.save(sess, pwc_ckpt_path)
                    break
                

if __name__ == '__main__':
    d = Dataset(num_clusters=20)
    d.read_train_data(dataset_fp='../../data/stackoverflow/train.csv', texts_vec_fp='../../data/stackoverflow/train_vec.txt')
    d.split_data_by_label()

    train(dataset=d)
