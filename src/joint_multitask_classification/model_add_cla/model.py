import os
import numpy as np

import tensorflow as tf

from utils import *


class AdaptiveMultiLabelsClassificationModel:
    def __init__(self,
                 max_sentence_len,
                 embedding_dim,
                 fc_dim,
                 num_clusters,
                 num_classes):
        self.max_sentence_len = max_sentence_len
        self.embedding_dim = embedding_dim
        self.fc_dim = fc_dim
        self.num_clusters = num_clusters
        self.num_classes = num_classes

        self.input_x = tf.placeholder(tf.float32, [None, self.max_sentence_len, embedding_dim], name='input_x')
        self.input_x_unlabeled = tf.placeholder(tf.float32, [None, self.max_sentence_len, embedding_dim], name='input_x_unlabeled')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.drop_keep_pro = tf.placeholder(tf.float32, name='drop_keep_pro')

        self.x = bn(self.input_x, scope='input_bn')
        self.x_unlabeled = bn(self.input_x_unlabeled, scope='input_bn')
        # I = W2*dropout(tanh(input*W1))
        # [batch_size, fc_dim]
        intent_representation, encode_info = self.representation_layer_by_cnn(self.x)
        intent_representation_unlabeled, encode_info_unlabeled = self.representation_layer_by_cnn(self.x_unlabeled)

        # autoencode task(cnn-based)
        cae_decode = self.cae_decoder(intent_representation, encode_info)
        cae_decode_unlabeled = self.cae_decoder(intent_representation_unlabeled, encode_info_unlabeled)

        # classification task
        w_cla = tf.get_variable(name='W_cla',
                                 shape=[self.fc_dim, self.num_classes],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        b_cla = tf.get_variable(name='b_cla',
                                 shape=[self.num_classes],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.nn.xw_plus_b(intent_representation, w_cla, b_cla, name='logits')
        scores = tf.nn.sigmoid(logits, name='sigmoid')

        # adaptive classification task
        w_ada_cla = tf.get_variable(name='W_ada_cla',
                                 shape=[self.fc_dim, self.num_clusters],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        b_ada_cla = tf.get_variable(name='b_ada_cla',
                                 shape=[self.num_clusters],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # [fc_dim, num_classes+num_clusters]
        w_all = tf.concat([w_cla, w_ada_cla], axis=1)
        b_all = tf.concat([b_cla, b_ada_cla], axis=0)
        logits_ada = tf.nn.xw_plus_b(intent_representation_unlabeled, w_all, b_all, name='logits_ada')
        scores_ada = tf.nn.sigmoid(logits_ada, name='sigmoid_ada')

        # find pseudo-label (sigmoid score > 0.5) for unlabeled data
        zeros_padding = tf.zeros_like(scores_ada, dtype=tf.float32)
        ones_padding = tf.ones_like(scores_ada, dtype=tf.float32)
        pseudo_label = tf.where(condition=tf.equal(scores_ada, tf.reduce_max(scores_ada, axis=1, keepdims=True)),
                                 x=ones_padding,
                                 y=zeros_padding)
        example_mask = tf.reduce_sum(pseudo_label, axis=1)
        zeros_padding_ = tf.zeros_like(example_mask, dtype=tf.float32)
        ones_padding_ = tf.ones_like(example_mask, dtype=tf.float32)
        self.example_mask = tf.where(condition=tf.greater_equal(example_mask, 1),
                x=ones_padding_,
                y=zeros_padding_)

        with tf.variable_scope('loss'):
            # ----autoencode loss----
            # cae_decode:[3, batch_size, max_seq_len, sentence_dim]
            self.loss_cae = tf.reduce_sum([tf.reduce_mean(tf.square(self.x-cae_decode_i), axis=0) for cae_decode_i in cae_decode])/3.0+\
                            tf.reduce_sum([tf.reduce_mean(tf.square(self.x_unlabeled-cae_decode_unlabeled_i), axis=0) for cae_decode_unlabeled_i in cae_decode_unlabeled])/3.0
            self.loss_cae /= 2.0

            # ----classification loss----
            losses_labeled = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            self.loss_labeled = tf.reduce_mean(losses_labeled, name='loss_labeled')
            
            # ----adaptive classification loss----
            losses_ada = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_ada, labels=pseudo_label)
            self.loss_ada = tf.reduce_mean(losses_ada, name='loss_ada')

            # ----add a class-embed-similarity loss----
            # [num_classes+num_clusters, num_classes+num_clusters]
            simi_mat = tf.matmul(a=tf.nn.l2_normalize(w_all, axis=0),
                                 b=tf.nn.l2_normalize(w_all, axis=0),
                                 transpose_a=True)
            simi_mat = (simi_mat+1)/2
            self.loss_simi = tf.reduce_mean(simi_mat, name='loss_simi')

            # ----pseudo-label min entropy loss----
            self.loss_minentro = -tf.reduce_mean(tf.reduce_sum(scores_ada*tf.log(scores_ada+1e-6), axis=1), name='loss_minentro')

            # self.loss = self.loss_cae + self.loss_labeled + self.loss_ada + self.loss_simi + self.loss_minentro
            self.loss = self.loss_cae

        with tf.variable_scope('train_op'):
            self.global_step = tf.Variable(0, name="global_step", trainable=True)
            self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss, global_step=self.global_step)
    
    def representation_layer_by_fc(self, input_x):
        input_x = tf.reduce_mean(input_x, axis=1)
        with tf.variable_scope('representation_layer', reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable(name='W1',
                                 shape=[self.embedding_dim, self.embedding_dim],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable(name='W2',
                                 shape=[self.embedding_dim, self.fc_dim],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            # I = W2*dropout(tanh(input*W1))
            # [batch_size, num_clusters]
            intent_representation = tf.matmul(tf.nn.dropout(tf.tanh(tf.matmul(input_x, w1)), keep_prob=self.drop_keep_pro), w2)
            self.sentence_embed = tf.identity(input=intent_representation, name='intent_representation')
            return self.sentence_embed

    def representation_layer_by_cnn(self, input_x, filter_sizes=[2, 3, 5], num_filters=64):
        # record info for reconstruct `input`
        encode_info = {}
        
        input_x = tf.expand_dims(input_x, -1)
        pooled_outputs = []
        with tf.variable_scope('representation_layer', reuse=tf.AUTO_REUSE):
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv1-%s" % filter_size, reuse=tf.AUTO_REUSE):
                    encode_vars, encode_shapes = [], []

                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_dim, 1, num_filters]
                    W = tf.get_variable('W',
                                        shape=filter_shape,
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable('b', shape=[num_filters], initializer=tf.contrib.layers.xavier_initializer())
                    conv = tf.nn.conv2d(
                        input_x,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h_1 = lrelu(bn(tf.nn.bias_add(conv, b), scope='bn-{}'.format(filter_size)))

                    encode_vars.append(W)
                    encode_shapes.append(input_x.get_shape().as_list())

                with tf.variable_scope("conv2-%s" % filter_size, reuse=tf.AUTO_REUSE):
                    # Convolution Layer
                    filter_shape = [self.max_sentence_len - filter_size + 1, 1, num_filters, num_filters]
                    W = tf.get_variable('W',
                                        shape=filter_shape,
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable('b', shape=[num_filters],
                                        initializer=tf.contrib.layers.xavier_initializer())
                    conv = tf.nn.conv2d(
                        h_1,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    # h_2:[1, 1, 1, num_filters]
                    h_2 = lrelu(bn(tf.nn.bias_add(conv, b), scope='bn-{}'.format(filter_size)))

                    encode_vars.append(W)
                    encode_shapes.append(h_1.get_shape().as_list())

                    encode_info[filter_size] = (encode_vars, encode_shapes)

                    pooled_outputs.append(h_2)
            # Combine all the conved features
            num_filters_total = num_filters * len(filter_sizes)
            h_cat = tf.concat(pooled_outputs, 3)
            w_fc = tf.get_variable(name='W_fc',
                                 shape=[num_filters_total, self.fc_dim],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            encode_info['fc'] = (w_fc, None)

            self.sentence_embed = lrelu(tf.matmul(a=tf.reshape(h_cat, [-1, num_filters_total]), b=w_fc), name='intent_representation')
            return self.sentence_embed, encode_info

    def cae_decoder(self, input_x, encode_info, filter_sizes=[2, 3, 5], num_filters=64):
        # input_x:[batch_size, fc_dim]
        # fc:[[num_filters_total, fc_dim]
        fc_vars, _ = encode_info['fc']
        out = lrelu(tf.matmul(a=input_x, b=fc_vars, transpose_b=True))

        # each element in out_splits's shape:[batch_size, num_filters]
        out_splits = tf.split(value=out,
                              num_or_size_splits=len(filter_sizes),
                              axis=1)
        out_splits = [tf.reshape(out_split, [-1, 1, 1, num_filters]) for out_split in out_splits]

        pred_lst = []
        for i, out_i in enumerate(out_splits):
            filter_size = filter_sizes[i]
            encode_vars, encode_shapes = encode_info[filter_size]
            for deconv_i, (encode_var, encode_shape) in enumerate(zip(encode_vars[::-1], encode_shapes[::-1])):
                out_i = lrelu(bn(
                tf.nn.conv2d_transpose(value=out_i,
                                       filter=encode_var,
                                       output_shape=tf.stack([tf.shape(out_i)[0], encode_shape[1], encode_shape[2], encode_shape[3]]),
                                       strides=[1, 1, 1, 1],
                                       padding='VALID'),
                    scope='de-bn-{}-{}'.format(filter_size, deconv_i)
                ))
            pred_lst.append(tf.squeeze(out_i, axis=[-1]))
        return pred_lst


