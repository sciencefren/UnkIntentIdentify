import numpy as np
import pandas as pd
import pickle


class Dataset:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def read_train_data(self, dataset_fp, texts_vec_fp):
        df = pd.read_csv(dataset_fp)
        labels = df['label_masked'].tolist()
        self.label2id_dct = dict(zip(set(labels)-{'unknown'}, range(len(set(labels))-1)))
        print(self.label2id_dct)
        self.num_classes = len(self.label2id_dct)
        self.unk_intent_id = self.num_classes
        self.label2id_dct['unknown'] = self.unk_intent_id
        print(self.label2id_dct)
        self.y = [self.label2id_dct[label] for label in labels]
        assert self.num_clusters >= len(self.label2id_dct), '`num_clusters` must greater or equal than `num_classes known`'

        with open(texts_vec_fp, 'r', encoding='utf-8') as f:
            texts_vec = []
            for line in f:
                text_vec = list(map(np.float32, line.split('\t')))
                texts_vec.append(text_vec)
        self.X = np.asarray(texts_vec, dtype=np.float32)

    def split_data_by_label(self):
        self.X_labeled, self.y_onehot = [], []
        self.X_unlabeled = []
        for i, label_id in enumerate(self.y):
            if label_id == self.unk_intent_id:
                self.X_unlabeled.append(self.X[i])
            else:
                self.X_labeled.append(self.X[i])
                y_tmp = np.zeros(self.num_classes, dtype=np.float32)
                y_tmp[label_id] = 1.0
                self.y_onehot.append(y_tmp)
        self.X_labeled, self.y_onehot, self.X_unlabeled = list(map(lambda x: np.asarray(x, dtype=np.float32),
                                                                   [self.X_labeled, self.y_onehot, self.X_unlabeled]))
        print('num labeled data:{}, num unlabeled data:{}'.format(self.X_labeled.shape[0], self.X_unlabeled.shape[0]))

    def gen_next_labeled_batch(self,
                       batch_size,
                       shuffle=True):
        steps = self.X_labeled.shape[0]//batch_size
        while True:
            if shuffle:
                idx = np.random.permutation(len(self.X_labeled))
                self.X_labeled = self.X_labeled[idx]
                self.y_onehot = self.y_onehot[idx]
            for step in range(steps):
                batch_x = self.X_labeled[step * batch_size:(step + 1) * batch_size, :]
                batch_y = self.y_onehot[step * batch_size:(step + 1) * batch_size, :]
                yield (batch_x, batch_y)

    def gen_next_unlabeled_batch(self,
                         batch_size,
                         shuffle=True):
        steps = self.X_unlabeled.shape[0]//batch_size
        while True:
            if shuffle:
                idx = np.random.permutation(len(self.X_unlabeled))
                self.X_unlabeled = self.X_unlabeled[idx]
            for step in range(steps):
                batch_x = self.X_unlabeled[step * batch_size:(step + 1) * batch_size, :]
                yield (batch_x, np.zeros((batch_size, self.num_clusters), dtype=np.float32))


if __name__ == '__main__':
    pass
