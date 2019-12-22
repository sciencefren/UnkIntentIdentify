import os
import numpy as np
import pandas as pd
import math
import pickle

import tensorflow as tf


class Model:
    def __init__(self, ckpt_fp):
        self.ckpt_fp = ckpt_fp

    def loadModel(self):
        g = tf.Graph()
        meta_fp = os.path.join(self.ckpt_fp, [fp for fp in os.listdir(self.ckpt_fp) if '.meta' in fp][0])
        with g.as_default():
            saver = tf.train.import_meta_graph(meta_fp)
        config = tf.ConfigProto(
            allow_soft_placement=True)
        self.sess = tf.Session(config=config, graph=g)
        checkpoint = tf.train.latest_checkpoint(self.ckpt_fp)
        if checkpoint:
            saver.restore(self.sess, checkpoint)
            print("[INFO] restore model from the checkpoint {0}".format(checkpoint))
        # input
        self.input = g.get_tensor_by_name('input_x:0')
        self.drop_keep_pro = g.get_tensor_by_name('drop_keep_pro:0')
        # output
        self.sentence_embed = g.get_tensor_by_name('representation_layer/intent_representation:0')
        self.scores = g.get_tensor_by_name('sigmoid:0')

    def predict(self, input_x):
        feed_dict = {self.input: input_x,
                self.drop_keep_pro: 1.0}
        scores = self.sess.run(self.scores, feed_dict=feed_dict)
        return scores

    def predict_on_file(self, dataset_fp, texts_vec_fp, label2y_fp, res_save_fp):
        df = pd.read_csv(dataset_fp)
        if os.path.exists(label2y_fp):
            with open(label2y_fp, 'rb') as f:
                label2id_dct = pickle.load(f)
                y2label_dct = {id_:label for label, id_ in label2id_dct.items()}
        else:
            return 

        pred, pred_flag = [], []
        with open(texts_vec_fp, 'r', encoding='utf-8') as f:
            for line in f:
                text_vec = np.asarray(list(map(np.float32, line.split('\t'))), dtype=np.float32).reshape(1, -1)
                scores = self.sess.run(self.scores, feed_dict={self.input: text_vec, self.drop_keep_pro: 1.0})
                scores_ranked = sorted(zip(range(len(y2label_dct)), scores[0]), key=lambda x:x[1], reverse=True)
                scores_filtered = '\t'.join([y2label_dct[y]+'-'+str(sco) for y, sco in scores_ranked if sco > 0.5])
                if not scores_filtered:
                    scores_filtered = y2label_dct[scores_ranked[0][0]]+'-'+str(scores_ranked[0][1])
                    pred_flag.append('top1')
                else:
                    pred_flag.append('multi_recall')
                pred.append(scores_filtered)
        df['pred']=pred
        df['pred_flag']=pred_flag
        df.to_csv(res_save_fp, index=False)


if __name__ == '__main__':
    m = Model('../../../../UnkIntentIdentifyModel/multilabels_cla_ckpt/')
    m.loadModel()

    m.predict_on_file(dataset_fp='../../../data/stackoverflow/train.csv',
            texts_vec_fp='../../../data/stackoverflow/train_vec.txt',
            label2y_fp='../../../data/stackoverflow/label2y_dct.pkl',
            res_save_fp='../../../data/stackoverflow/pred_multilabels_model.csv')
