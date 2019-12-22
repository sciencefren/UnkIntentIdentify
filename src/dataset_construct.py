import os
import numpy as np
import pandas as pd
import pickle
from time import time

from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, './')
sys.path.insert(0, '../../pretrain_models/bert_en')
from sentence_embed.SentenceEmbedModel import SentenceEmbedModel


class Dataset:

    def read_from_csv(self, dataset_fp):
        df = pd.read_csv(dataset_fp)
        self.texts = df['title'].values
        self.labels = df['label'].values

        self.num_classes = len(set(self.labels))
        self.label2id_dct = dict(zip(list(set(self.labels)) + ['unknown'], range(self.num_classes + 1)))

    def split_data(self,
                   num_unk_classes=5,
                   labeled_ratio=0.1,
                   save_dir=''):
        skf = StratifiedKFold(n_splits=self.num_classes // 2, random_state=6)
        for tra_idx, tst_idx in skf.split(self.texts, self.labels):
            tra_texts, tst_texts = self.texts[tra_idx], self.texts[tst_idx]
            tra_labels, tst_labels = self.labels[tra_idx], self.labels[tst_idx]
            print('----------------------')
            print(pd.Series(tst_labels).value_counts())
            break

        unk_classes = np.random.choice(list(set(self.label2id_dct.keys())-{'unknown'}),
                                       size=num_unk_classes,
                                       replace=False)
        print('unknown intent set:{}'.format(unk_classes))

        tra_labels_masked = []
        for label in tra_labels:
            if label in unk_classes:
                label = 'unknown'
            elif np.random.rand() > labeled_ratio:
                label = 'unknown'
            else:
                pass
            tra_labels_masked.append(label)

        # print('------')
        # print(len(set(tra_labels_masked)))
        # print(pd.Series(tra_labels_masked).value_counts())

        df_tra = pd.DataFrame({'text': list(tra_texts),
                               'label': tra_labels,
                               'label_masked': tra_labels_masked})
        df_tra.to_csv(os.path.join(save_dir, 'train.csv'), index=False)

        df_tst = pd.DataFrame({'text': list(tst_texts),
                               'label': tst_labels})
        df_tst.to_csv(os.path.join(save_dir, 'test.csv'), index=False)

        self.convert_text_to_vector(tra_texts, os.path.join(save_dir, 'train_vec.txt'))

        self.convert_text_to_vector(tst_texts, os.path.join(save_dir, 'test_vec.txt'))

    def convert_text_to_vector(self,
                               texts,
                               save_fp):
        model_dir = '../../pretrain_models/bert_en/bert_uncased_en/'
        vocab_fp = os.path.join(model_dir, 'vocab.txt')
        model_config_fp = os.path.join(model_dir, 'bert_config.json')
        model_init_ckpt_fp = os.path.join(model_dir, 'checkpoint')

        sentence_embed_model = SentenceEmbedModel(vocab_fp=vocab_fp,
                                                  model_config_fp=model_config_fp,
                                                  model_init_ckpt_fp=model_init_ckpt_fp)
        sentence_embed_model.load_model()
        
        f = open(save_fp, 'w', encoding='utf-8')
        for i, text in enumerate(texts):
            t0 = time()
            # while `method='sequence'`
            # return text_vec and true_length
            # text_vec is ahpe:[true_length, embedding_dim]
            text_vec, true_length = sentence_embed_model.get_sentence_embed(text, method='sequence')
            line = str(true_length)+'\t'+'\t'.join(list(map(str, text_vec.reshape(1, -1)[0])))
            f.writelines(line+'\n')
            print('text {}/{}, tokens size {}, write to file use time:{:.4}'.format(i+1, len(texts), true_length, time()-t0))
        f.close()
        sentence_embed_model.finish()
        print('model use session closed ...')


if __name__ == '__main__':
    d = Dataset()
    d.read_from_csv('../data/stackoverflow/stackoverflow_dataset.csv')

    d.split_data(save_dir='../data/stackoverflow/')
