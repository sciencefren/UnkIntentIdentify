import os
import numpy as np
from time import time

import tensorflow as tf

from dataset import Dataset
from model import AdaptiveMultiLabelsClassificationModel as Model


def train(dataset):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = Model(embedding_dim=768,
                        fc_dim=256,
                        max_sentence_len=dataset.max_sentence_len,
                        num_classes=d.num_classes,
                        num_clusters=8)
            # check variables
            for v in tf.trainable_variables():
                #if 'intent' in v.name:
                print(v.name)

            pwc_ckpt_path = '../../../../UnkIntentIdentifyModel/multilabels_ada_cla_ckpt/'
            checkpoint = tf.train.latest_checkpoint(pwc_ckpt_path)
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)

            sess.run(tf.global_variables_initializer())
            if checkpoint:
                saver.restore(sess, checkpoint)

            batch_size = 64
            print_info_steps = 10
            svae_ckpt_steps = 500
            while True:
                t0 = time()
                (batch_x, batch_y) = next(dataset.gen_next_labeled_batch(batch_size=batch_size))
                (batch_x_unlabeled, _) = next(dataset.gen_next_unlabeled_batch(batch_size=2*batch_size))
                #print('fetch batch data {} use time {:.4}'.format(batch_size, time()-t0))
                t0 = time()
                _, loss, loss_labeled, loss_ada, loss_simi, loss_minentro, example_mask, step = sess.run([model.train_op,
                    model.loss,
                    model.loss_labeled,
                    model.loss_ada,
                    model.loss_simi,
                    model.loss_minentro,
                    model.example_mask,
                    model.global_step],
                                             feed_dict={model.input_x: batch_x,
                                                        model.input_x_unlabeled: batch_x_unlabeled,
                                                        model.input_y: batch_y,
                                                        model.drop_keep_pro: 0.6})
                #print('run one step use time {:.4}'.format(time()-t0))
                if not (step+1)%print_info_steps:
                    print("[INFO] step:{}, ada pro:{:.4}, loss:{:.4}, loss_labeled:{:.4}, loss_ada:{:.4}, loss_simi:{:.4}, loss_loss_minentro:{:.4}".format(step,
                        example_mask.sum()/(2*batch_size),
                        loss,
                        loss_labeled,
                        loss_ada,
                        loss_simi,
                        loss_minentro))
                if step > 10000 and loss < 0.2:
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
