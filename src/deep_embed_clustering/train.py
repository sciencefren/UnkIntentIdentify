import os
import numpy as np

import tensorflow as tf

from model import DEC
from dataset import Dataset

def train(dataset, batch_size, result_save_fp):
    """

    :param dataset: a class has attributes `X` and `y`
                    `X` is the whole dataset's sentence embedding
    :return:
    """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model = DEC(params={
        "dataset": dataset,
        "feature_dims": dataset.X_intent_enhenced.shape[1],
        "n_clusters": 20,
        "alpha": 1.0
    })
    
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)
    
    # phase 1: parameter initialization
    # use pretrain-lm `BERT`

    # phase 2: parameter optimization
    dec_ckpt_path = os.path.join('../../../UnkIntentIdentifyModel/dec_ckpt', 'model.ckpt')
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        # initialize mu
        assign_mu_op = model.get_assign_cluster_centers_op(dataset.X_intent_enhenced)
        _ = sess.run(assign_mu_op)

        for cur_epoch in range(100):
            q = sess.run(model.q, feed_dict={model.z: dataset.X_intent_enhenced,
                                             model.input_batch_size: dataset.X.shape[0]})
            p = model.target_distribution(q)
            
            # per one epoch
            for iter_, (batch_x, batch_p, batch_y) in enumerate(dataset.gen_next_batch(x=dataset.X_intent_enhenced,
                                                                                       y=p,
                                                                                       label=dataset.y,
                                                                                       batch_size=batch_size,
                                                                                       epoches=4)):
                _, loss, pred = sess.run([model.optimizer, model.loss, model.pred],
                                   feed_dict={model.z: batch_x,
                                              model.p: batch_p,
                                              model.input_batch_size: batch_x.shape[0]})
                print("[DEC] epoch: {}, loss: step: {}, {:.4}, acc: {:.4}".format(cur_epoch, iter_, loss, model.cluster_acc(batch_y, pred)))
                saver.save(sess, dec_ckpt_path)
        pred = sess.run(model.pred, feed_dict={model.z: dataset.X_intent_enhenced,
                                              model.input_batch_size: dataset.X.shape[0]})
        df_res = dataset.df
        df_res['pred'] = [pred_i for pred_i in pred]
        df_res.to_csv(result_save_fp, index=False)
    
    
if __name__=="__main__":
    d = Dataset()
    d.read_train_data(dataset_fp='../../data/stackoverflow/train.csv',
                      texts_vec_fp='../../data/stackoverflow/train_vec.txt',
                      label2y_save_fp='../../data/stackoverflow/label2y_dct.pkl')
    d.get_intent_enhenced_embed(pwc_ckpt_fp='../../../UnkIntentIdentifyModel/pwc_mt_ckpt')

    train(d, 256, '../../data/stackoverflow/result/result_pwcmt_pooled.csv')
