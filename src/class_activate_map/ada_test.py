import os
import numpy as np
import pickle

from ClassActivateMap import ClassActivateMap

import sys
from visual import cam_visualization
sys.path.insert(0, '../../../pretrain_models/bert_en')
import tokenization

sys.path.insert(0, '../')
from sentence_embed.SentenceEmbedModel import SentenceEmbedModel


class TextProcessor:
    def __init__(self,
                 vocab_fp,
                 do_lower_case=True):
        self.vocab_fp = vocab_fp
        self.do_lower_case = do_lower_case

    def load_model(self):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_fp,
                                                    do_lower_case=self.do_lower_case)

    def get_tokens(self,
                   text,
                   max_seq_length):
        """Converts a single `text` into a single `Standard-Input`."""

        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        return tokens


if __name__ == '__main__':

    #加载句向量表征模型
    model_dir = '../../../pretrain_models/bert_en/bert_uncased_en/'
    vocab_fp = os.path.join(model_dir, 'vocab.txt')
    model_config_fp = os.path.join(model_dir, 'bert_config.json')
    model_init_ckpt_fp = os.path.join(model_dir, 'checkpoint')

    sentence_embed_model = SentenceEmbedModel(vocab_fp=vocab_fp,
                                              model_config_fp=model_config_fp,
                                              model_init_ckpt_fp=model_init_ckpt_fp)
    sentence_embed_model.load_model()

    # 加载获取文本tokens的类
    tokenizer = TextProcessor(vocab_fp)
    tokenizer.load_model()

    #加载label2y dict
    label2y_dct_fp = '../../data/stackoverflow/label2y_dct.pkl'
    with open(label2y_dct_fp, 'rb') as f:
        label2y_dct = pickle.load(f)
    for i in range(len(label2y_dct), len(label2y_dct)+8):
        label2y_dct[i] = i
    print(label2y_dct)

    #构建ClassActivateMap类初始化参数
    input_tensor_name_lst = ['input_x_unlabeled:0', 'drop_keep_pro:0']
    conv_name2filtersize_dct = dict([('representation_layer_1/conv-maxpool-{}/relu:0'.format(filtersize), filtersize) for filtersize in [2, 3, 5]])
    logits_name = 'logits_ada:0'
    #ClassActivateMap初始化
    cam = ClassActivateMap(ckpt_fp='../../temp_model/multilabels_ada_cla_ckpt_new/',
                           label2id_dct=label2y_dct,
                           input_tensor_name_lst=input_tensor_name_lst,
                           conv_layer_tensor_name2filter_size_dct=conv_name2filtersize_dct,
                           logits_tensor_name=logits_name)
    #输入文本
    text = 'How do you stop a Visual Studio generated web service proxy class from encoding?'
    #
    max_sentence_len = 68
    #
    tokens = tokenizer.get_tokens(text, max_sentence_len)
    print(tokens)

    #构建输入值，为字级别输入序列
    text_vec, true_length = sentence_embed_model.get_sentence_embed(text)
    pad_vec = np.zeros((max_sentence_len - true_length, 768), dtype=np.float32)
    text_vec_padded = np.concatenate((text_vec, pad_vec), axis=0).reshape(1, max_sentence_len, 768)
    #
    input_value_lst = [text_vec_padded, 1.0]
    #获取cam结果
    res_cam = cam.get_text_final_cam(input_value_lst, 18, max_sentence_len, true_length)
    print(res_cam)
    #可视化，因示例任务的输入包含起始符<bos>和终止符<eos>，所以分别在前后加上加上
    cam_visualization(res_cam, text_fragment_lst=tokens)
