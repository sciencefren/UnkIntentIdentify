import os
import numpy as np
import tensorflow as tf


class ClassActivateMap:
    def __init__(self,
                 ckpt_fp,
                 label2id_dct,
                 input_tensor_name_lst,
                 conv_layer_tensor_name2filter_size_dct,
                 logits_tensor_name=''):
        #
        self.label2id_dct = label2id_dct
        self.id2label_dct = {id_: label_ for label_, id_ in self.label2id_dct.items()}
        self.input_tensor_name_lst = input_tensor_name_lst
        self.logits_tensor_name = logits_tensor_name
        self.conv_layer_tensor_name2filter_size_dct = conv_layer_tensor_name2filter_size_dct
        #找到.meta文件
        meta_fp = os.path.join(ckpt_fp, [fp for fp in os.listdir(ckpt_fp) if '.meta' in fp][0])
        #创建一个图
        tf.reset_default_graph()
        self.g = tf.Graph()
        with self.g.as_default():
            saver = tf.train.import_meta_graph(meta_fp)
        #设置session的配置
        config = tf.ConfigProto(device_count={"CPU": 1},
                                inter_op_parallelism_threads=6,
                                intra_op_parallelism_threads=6,
                                log_device_placement=False)
        self.sess = tf.Session(graph=self.g, config=config)
        with self.sess.as_default(), self.g.as_default():
            checkpoint = tf.train.latest_checkpoint(ckpt_fp)
            #恢复模型
            if checkpoint:
                saver.restore(self.sess, checkpoint)
                print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            # 定义输入接口
            self.input_tensor_lst = []
            for tensor_name in self.input_tensor_name_lst:
                self.input_tensor_lst.append(self.g.get_tensor_by_name(tensor_name))
            #定义输出接口
            self.logits = self.g.get_tensor_by_name(self.logits_tensor_name)

    def get_conv_out(self, conv_i_name):
        #获取CNN中的filter输出
        return self.g.get_tensor_by_name(conv_i_name)

    def get_single_grad_cam(self, label_id, conv_i_name):
        #
        with self.g.as_default():
            #构建一个mask用于获取指定类别的logits值
            mask = tf.one_hot(label_id, depth=len(self.label2id_dct))
            # 0-dim, a scalar
            y_c = tf.reduce_sum(tf.multiply(mask, self.logits))
            # [batch_size, max_sentence_len, 1(input_channels), num_filters], 即logits对CNN中每个filter元素的导数值
            conv_out = self.get_conv_out(conv_i_name)
            grads = tf.gradients(y_c, conv_out)[0]
            # [batch_size, max_sentence_len, 1(input_channels), num_filters]
            grads_norm = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
            # squeeze axis=0 because batch_size=1, axis=2 because input_channels=1
            # (num_filters,) 对应着CNN中每个filter map的重要性
            alpha_c_k = tf.reduce_mean(tf.squeeze(grads_norm, axis=[0, 2]), axis=0)
            # [batch_size, max_sentence_len, 1] through boardcasting
            L_gradcam = tf.nn.relu(tf.reduce_sum(tf.multiply(conv_out, alpha_c_k), axis=-1))
            #最大值归一化
            L_gradcam = tf.div(L_gradcam, tf.reduce_max(L_gradcam))
        return tf.squeeze(L_gradcam)


    def get_rank_label(self, feed_dict):
        #获取最大logits
        score = self.sess.run(self.logits, feed_dict)
        score_label_tpl = sorted(zip(score[0], list(self.label2id_dct.keys())), key=lambda x: x[0], reverse=True)
        return score_label_tpl[0][1]


    def get_text_final_cam(self, input_value_lst, target_label, max_sentence_len, true_length):
        #构造输入feed_dict
        feed_dict = dict(zip(self.input_tensor_name_lst, input_value_lst))
        #确定label id
        if target_label=='top':
            label = self.get_rank_label(feed_dict)
        else:
            label = target_label
        print('the selected label is: {}'.format(label))
        #用于累积每个filter map给出的文本片段重要性
        cam = np.zeros(max_sentence_len, dtype=np.float32)
        #用于累积每个文本片段被这些filter map卷积过多少次，该值后面用于归一化
        position_vote_num = np.zeros(max_sentence_len, dtype=np.float32)
        for conv_i_name, filter_size in self.conv_layer_tensor_name2filter_size_dct.items():
            cam_filter_i = self.sess.run(self.get_single_grad_cam(self.label2id_dct[label], conv_i_name), feed_dict=feed_dict)
            for i, activation_i in enumerate(cam_filter_i):
                cam[i:i + filter_size] += activation_i
                position_vote_num[i:i + filter_size] += 1
        #卷积数归一化
        cam /= position_vote_num
        #最大值归一化
        cam /= cam.max()
        return cam[:true_length].reshape(1, -1)
