import os
import numpy as np

import tensorflow as tf


class MultiLabelsClassificationModel:
    def __init__(self,
                 max_sentence_len,
                 embedding_dim,
                 num_clusters,
                 num_classes):
        self.max_sentence_len = max_sentence_len
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.num_classes = num_classes

        self.input_x = tf.placeholder(tf.float32, [None, self.max_sentence_len, embedding_dim], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.drop_keep_pro = tf.placeholder(tf.float32, name='drop_keep_pro')

        # I = W2*dropout(tanh(input*W1))
        # [batch_size, num_clusters]
        intent_representation = self.representation_layer_by_cnn(self.input_x)

        # classification task
        w_cla = tf.get_variable(name='W_cla',
                                 shape=[self.num_clusters, self.num_classes],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        b_cla = tf.get_variable(name='b_cla',
                                 shape=[self.num_classes],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.nn.xw_plus_b(intent_representation, w_cla, b_cla, name='logits')
        scores = tf.nn.sigmoid(logits, name='sigmoid')

        with tf.variable_scope('loss'):
            # ----classification loss----
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
        
        with tf.variable_scope('train_op'):
            self.global_step = tf.Variable(0, name="global_step", trainable=True)
            self.train_op = tf.train.GradientDescentOptimizer(3e-4).minimize(self.loss, global_step=self.global_step)
    
    def representation_layer_by_fc(self, input_x):
        input_x = tf.reduce_mean(input_x, axis=1)
        with tf.variable_scope('representation_layer', reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable(name='W1',
                                 shape=[self.embedding_dim, self.embedding_dim],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable(name='W2',
                                 shape=[self.embedding_dim, self.num_clusters],
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
            self.sentence_embed = tf.reshape(h_pool, [-1, num_filters_total], name='intent_representation')
