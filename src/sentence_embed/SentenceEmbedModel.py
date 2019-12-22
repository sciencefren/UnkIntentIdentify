# coding=utf-8
import os
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, '../../../pretrain_models/bert_en')
import modeling
import tokenization

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

class TextProcessor:
    def __init__(self,
                 vocab_fp,
                 do_lower_case=True):
        self.vocab_fp = vocab_fp
        self.do_lower_case = do_lower_case

    def load_model(self):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_fp,
                                                    do_lower_case=self.do_lower_case)

    def convert_single_example(self,
                               text,
                               max_seq_length):
        """Converts a single `text` into a single `Standard-Input`."""

        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        segment_ids = [0]*len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        return map(lambda x: np.asarray(x, dtype=np.int32).reshape(1, -1), [input_ids, input_mask, segment_ids])


class SentenceEmbedModel:
    def __init__(self,
                 vocab_fp,
                 model_config_fp,
                 model_init_ckpt_fp):
        #
        self.vocab_fp = vocab_fp
        #
        self.model_config_fp = model_config_fp
        self.model_init_ckpt_fp = model_init_ckpt_fp

    def load_model(self):
        # load tokenizer
        self.tokenizer = TextProcessor(self.vocab_fp)
        self.tokenizer.load_model()

        #
        model_config = modeling.BertConfig.from_json_file(self.model_config_fp)
        #
        self.input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
        #
        model = modeling.BertModel(
            config=model_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        # restore model
        tvars = tf.trainable_variables()
        (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.model_init_ckpt_fp)
        tf.train.init_from_checkpoint(self.model_init_ckpt_fp, assignment)
        print('restore bert model ...')
        # get the last layer output embedding
        # [1, max_seq_length, embedding_dim]
        self.encoder_last_layer = model.get_sequence_output()

        # create a session
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)
        self.sess = tf.Session(config=session_conf)
        self.sess.run(tf.global_variables_initializer())



    def get_sentence_embed(self,
                           text,
                           max_seq_length=512,
                           method='sequence'):
        """

        :param text: a sentence
        :param max_seq_length: the max length of tokens
        :param method: the form of sentence vector returned
                       `=sequence` return all tokens embedding, shape:[max_seq_length, embedding_dim]
                       `=pooled` return the average of all tokens' embedding, shape:[1, embedding_dim]
                       `=first` return the '[CLS]' token's embedding, shape:[1, embedding_dim]
        :return:
        """
        assert method in {'sequence', 'pooled', 'first'}, 'input params `method` is error'

        input_ids, input_mask, segment_ids = self.tokenizer.convert_single_example(text,
                                                                                   max_seq_length)
        #
        feed_dict = dict(zip([self.input_ids, self.input_mask, self.segment_ids],
                             [input_ids, input_mask, segment_ids]))
        sentence_embed = self.sess.run(self.encoder_last_layer, feed_dict=feed_dict)

        if method == 'sequence':
            true_length = input_mask.sum()
            return sentence_embed[:, :true_length, :][0], true_length
        elif method == 'pooled':
            return np.mean(sentence_embed[0], axis=0, keepdims=True)
        else:
            return sentence_embed[0][0:1, :]
    
    def finish(self):
        self.sess.close()


def compute_similarity(sent_embed_a, sent_embed_b):
    """

    :param sent_embed_a: the shape is [1, embedding_dim]
    :param sent_embed_b:
    :return:
    """
    sent_embed_a /= np.sqrt(np.sum(np.square(sent_embed_a)))
    sent_embed_b /= np.sqrt(np.sum(np.square(sent_embed_b)))

    return np.sum(sent_embed_a*sent_embed_b)


if __name__ == '__main__':
    model_dir = '../../../pretrain_models/bert_en/bert_uncased_en/'
    vocab_fp = os.path.join(model_dir, 'vocab.txt')
    model_config_fp = os.path.join(model_dir, 'bert_config.json')
    model_init_ckpt_fp = os.path.join(model_dir, 'checkpoint')

    sentence_embed_model = SentenceEmbedModel(vocab_fp=vocab_fp,
                                              model_config_fp=model_config_fp,
                                              model_init_ckpt_fp=model_init_ckpt_fp)
    sentence_embed_model.load_model()

    # validate some case through text-similarity
    text_a = 'Can I create a Visual Studio macro to launch a specific project in the debugger?'
    text_b = 'Advantages of VS 2008 over VS 2005'
    text_c = 'How can I retore svn control if the .svn folder has been damaged'

    embed_a = sentence_embed_model.get_sentence_embed(text_a, method='pooled')
    embed_b = sentence_embed_model.get_sentence_embed(text_b, method='pooled')
    embed_c = sentence_embed_model.get_sentence_embed(text_c, method='pooled')

    print('(a, b):{:.4}'.format(compute_similarity(embed_a, embed_b)))
    print('(b, c):{:.4}'.format(compute_similarity(embed_b, embed_c)))

