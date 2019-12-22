import numpy as np
import pickle

from ClassActivateMap import ClassActivateMap
from visual import cam_visualization

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


if __name__ == '__main__':
    #定义输入辅助类
    max_sentence_len = 40
    pdh = PreDataHelper('../../model/unknown_model/unkVSres_char2ind_dct.pkl',
                        max_sentence_len)
    #构建ClassActivateMap类初始化参数
    input_tensor_name_lst = ['input_text:0', 'input_sentence_len:0', 'dropout_keep_prob:0']
    conv_name2filtersize_dct = dict([('conv-maxpool-{}/relu:0'.format(filtersize), filtersize) for filtersize in [1, 2, 3]])
    logits_name = 'output/logit:0'
    #ClassActivateMap初始化
    cam = ClassActivateMap(ckpt_fp='../../model/unknown_model/checkpoints_unkVSres_transformer_01',
                           label2id_dct={'unknown':0, 'known':1},
                           input_tensor_name_lst=input_tensor_name_lst,
                           conv_layer_tensor_name2filter_size_dct=conv_name2filtersize_dct,
                           logits_tensor_name=logits_name,)
    #输入文本
    text = '北京哪里好玩呢'
    #构建输入值，为字级别输入序列
    sentence_ids, true_length = pdh.char2id(text)
    input_value_lst = [sentence_ids, true_length, 1.0]
    #获取cam结果
    res_cam = cam.get_text_final_cam(input_value_lst, 'top', max_sentence_len, true_length[0])
    print(res_cam)
    #可视化，因示例任务的输入包含起始符<bos>和终止符<eos>，所以分别在前后加上加上
    cam_visualization(res_cam, text_fragment_lst=['<bos>']+list(text)+['<eos>'])
