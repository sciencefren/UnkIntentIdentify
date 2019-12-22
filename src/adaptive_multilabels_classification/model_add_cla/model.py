import os
import numpy as np

import tensorflow as tf


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

        # I = W2*dropout(tanh(input*W1))
        # [batch_size, fc_dim]
        intent_representation = self.representation_layer_by_cnn(self.input_x)
        intent_representation_unlabeled = self.representation_layer_by_cnn(self.input_x_unlabeled)

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

            self.loss = self.loss_labeled + self.loss_ada + self.loss_simi + self.loss_minentro

        with tf.variable_scope('train_op'):
            self.global_step = tf.Variable(0, name="global_step", trainable=True)
            self.train_op = tf.train.AdamOptimizer(3e-4).minimize(self.loss, global_step=self.global_step)
    
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
        input_x = tf.expand_dims(input_x, -1)
        pooled_outputs = []
        with tf.variable_scope('representation_layer', reuse=tf.AUTO_REUSE):
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=tf.AUTO_REUSE):
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
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.max_sentence_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            w_fc = tf.get_variable(name='W_fc',
                                 shape=[num_filters_total, self.fc_dim],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            self.sentence_embed = tf.nn.relu(tf.matmul(a=tf.reshape(h_pool, [-1, num_filters_total]), b=w_fc), name='intent_representation')
            return self.sentence_embed
