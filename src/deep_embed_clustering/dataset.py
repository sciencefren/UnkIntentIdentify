import os
import numpy as np
import pandas as pd
import math
import pickle

import tensorflow as tf


class Dataset:
    def read_train_data(self, dataset_fp, texts_vec_fp, label2y_save_fp):
        df = pd.read_csv(dataset_fp)
        labels = df['label'].tolist()
        self.label2id_dct = dict(zip(set(labels), range(len(set(labels)))))
        self.y = np.asarray([self.label2id_dct[label] for label in labels], dtype=np.int32)
        with open(label2y_save_fp, 'wb') as f:
            pickle.dump(self.label2id_dct, f)

        with open(texts_vec_fp, 'r', encoding='utf-8') as f:
            texts_vec = []
            for line in f:
                text_vec = list(map(np.float32, line.split('\t')))
                texts_vec.append(text_vec)
        self.X = np.asarray(texts_vec, dtype=np.float32)
        self.df = df

    def get_intent_enhenced_embed(self, pwc_ckpt_fp):
        g = tf.Graph()
        meta_fp = os.path.join(pwc_ckpt_fp, [fp for fp in os.listdir(pwc_ckpt_fp) if '.meta' in fp][0])
        with g.as_default():
            saver = tf.train.import_meta_graph(meta_fp)
        config = tf.ConfigProto(
            allow_soft_placement=True)
        sess = tf.Session(config=config, graph=g)
        checkpoint = tf.train.latest_checkpoint(pwc_ckpt_fp)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore [PWC] model from the checkpoint {0}".format(checkpoint))
        # pwc input
        pwc_input = g.get_tensor_by_name('input_x:0')
        pwc_drop_keep_pro = g.get_tensor_by_name('drop_keep_pro:0')
        # pwc output
        pwc_output = g.get_tensor_by_name('representation_layer/intent_representation:0')

        self.X_intent_enhenced = []
        for x in self.X:
            x_intent_enhenced = sess.run(pwc_output, feed_dict={pwc_input: x.reshape(1, -1), pwc_drop_keep_pro: 1.0})
            self.X_intent_enhenced.append(x_intent_enhenced[0])
        sess.close()
        self.X_intent_enhenced = np.asarray(self.X_intent_enhenced, dtype=np.float32)
        print('get intent enhenced representation finished.')

    def gen_next_batch(self,
                       x,
                       y,
                       label,
                       batch_size,
                       shuffle=True,
                       epoches=None):
            
        assert len(x) >= batch_size, "batch size must be smaller than data size {}.".format(len(x))
        
        if epoches:
            steps = len(x)//batch_size
        else:
            assert False, "epoch or iteration must be set."

        for _ in range(epoches):
            if shuffle:
                idx = np.random.permutation(len(x))
                x = x[idx]
                y = y[idx]
                label = label[idx]
            for step in range(steps):
                batch_x = x[step*batch_size:(step+1)*batch_size, :]
                batch_y = y[step*batch_size:(step+1)*batch_size, :]
                batch_z = label[step*batch_size:(step+1)*batch_size]
                yield (batch_x, batch_y, batch_z)


if __name__ == '__main__':
    pass
