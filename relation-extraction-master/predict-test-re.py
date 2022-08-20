import time
import json
import os
import codecs
import ast
import torch
from torch import nn
from tqdm import tqdm
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertTokenizerFast

def train_data_processing(tokenizer_fast,
                          train_raw_path='data_raw/train.conll_convert.conll',
                          train_path='data_relation/train.json'):
    # 构造训练集样本
    with open(train_raw_path, 'r', encoding='utf-8') as f:
        relation_sentences = f.read().split('\n\n')

    train_data = list()
    for sentence_id, relation_sentence in tqdm(enumerate(relation_sentences[:-1])):
        ners_sentence = dict()
        relation_dict = dict()  # 记录属性集合
        for sentence in relation_sentence.split('\n'):
            try:
                ners_sentence = json.loads(sentence)
            except Exception as e:
                sub, obj, relation_type = sentence.split('\t')
                if (sub != '[]') and (obj != '[]'):
                    relation_dict[f'{sub}\t{obj}'] = relation_type

        outputs = tokenizer_fast(ners_sentence['sent'], return_offsets_mapping=True)
        offset_mapping = []
        for i, (start, end) in enumerate(outputs["offset_mapping"]):
            if (end > 0) and (i >= 2):
                start -= (i - 1)
                end -= (i - 1)
            offset_mapping.append((start, end))

        ner_idx_dict = dict()
        for ent in ners_sentence['ners']:
            ent_start, ent_end, ent_type, ent_text = ent[0], ent[1], ent[2], ent[3]

            # 构建ent2token_spans
            ent_start_token_idx, ent_end_token_idx = -1, -1
            for idx, (span_start, span_end) in enumerate(offset_mapping):
                if span_end == 0:
                    continue

                if span_start == ent_start:
                    ent_start_token_idx = idx

                if span_end == ent_end:
                    ent_end_token_idx = idx
                    break

            if (ent_start_token_idx == -1) or (ent_end_token_idx == -1):
                print(f'{ent_text}无对应token')

            ner_idx_dict[str(ent)] = [ent_start_token_idx, ent_end_token_idx]

        ner_token_idx_dict = dict()
        sentence_text = '[CLS] ' + ners_sentence['sent'] + ' [SEP]'
        position_ids = [i for i in range(len(offset_mapping))]
        for ner in ners_sentence['ners']:
            sentence_text += f' [{ner[2]}] [/{ner[2]}]'
            ner_token_idx_dict[str(ner)] = [len(position_ids)+i for i in range(2)]
            position_ids.extend(ner_idx_dict[str(ner)])

        # 构造一条训练集样本
        relation, relation_label, relation_idx = [], [], []
        for sub in ners_sentence['ners']:
            for obj in ners_sentence['ners']:
                if str(sub) == str(obj):
                    continue
                relation.append(f'{str(sub)}\t{str(obj)}')
                relation_label.append(relation_dict.get(f'{str(sub)}\t{str(obj)}', '无'))
                relation_idx.append(ner_token_idx_dict[str(sub)]+ner_token_idx_dict[str(obj)])

        train_data.append({'sentence_text': sentence_text,
                           'position_ids': position_ids,
                           'relation': relation,
                           'relation_label': relation_label,
                           'relation_idx': relation_idx})

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

def test_data_processing(tokenizer_fast, data_path, test_path='data_relation/testB.json'):
    # 构造测试集样本
    with open(data_path, 'r', encoding='utf-8') as f:
        relation_sentences = f.read().split('\n\n')

    test_data = list()
    for ners_sentence in tqdm(relation_sentences[:-1]):
        ners_sentence = json.loads(ners_sentence)
        relation_dict = dict()

        outputs = tokenizer_fast(ners_sentence['sent'], return_offsets_mapping=True)
        offset_mapping = []
        for i, (start, end) in enumerate(outputs["offset_mapping"]):
            if (end > 0) and (i >= 2):
                start -= (i - 1)
                end -= (i - 1)
            offset_mapping.append((start, end))

        ner_idx_dict = dict()
        for ent in ners_sentence['ners']:
            ent_start, ent_end, ent_type, ent_text = ent[0], ent[1], ent[2], ent[3]

            # 构建ent2token_spans
            ent_start_token_idx, ent_end_token_idx = -1, -1
            for idx, (span_start, span_end) in enumerate(offset_mapping):
                if span_end == 0:
                    continue

                if span_start == ent_start:
                    ent_start_token_idx = idx

                if span_end == ent_end:
                    ent_end_token_idx = idx
                    break

            if (ent_start_token_idx == -1) or (ent_end_token_idx == -1):
                print(f'{ent_text}无对应token')

            ner_idx_dict[str(ent)] = [ent_start_token_idx, ent_end_token_idx]

        ner_token_idx_dict = dict()
        sentence_text = '[CLS] ' + ners_sentence['sent'] + ' [SEP]'
        position_ids = [i for i in range(len(offset_mapping))]
        for ner in ners_sentence['ners']:
            sentence_text += f' [{ner[2]}] [/{ner[2]}]'
            ner_token_idx_dict[str(ner)] = [len(position_ids)+i for i in range(2)]
            position_ids.extend(ner_idx_dict[str(ner)])

        # 构造一条训练集样本
        relation, relation_label, relation_idx = [], [], []
        for sub in ners_sentence['ners']:
            for obj in ners_sentence['ners']:
                if str(sub) == str(obj):
                    continue
                relation.append(f'{str(sub)}\t{str(obj)}')
                relation_label.append(relation_dict.get(f'{str(sub)}\t{str(obj)}', '无'))
                relation_idx.append(ner_token_idx_dict[str(sub)]+ner_token_idx_dict[str(obj)])

        test_data.append({'sent': ners_sentence['sent'],
                           'ners': ners_sentence['ners'],
                           'sentence_text': sentence_text,
                           'position_ids': position_ids,
                           'relation': relation,
                           'relation_label': relation_label,
                           'relation_idx': relation_idx})

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

class CustomModel(BertPreTrainedModel):
    """自定义模型"""

    def __init__(self, config, use_rope=True):
        super(CustomModel, self).__init__(config)
        self.bert = BertModel(config)  # transformers的写法，方便保存，加载模型

        self.linear_dim = int(config.hidden_size / config.num_attention_heads)  # 默认
        self.linear = nn.Linear(config.hidden_size, self.linear_dim * 2)

        self.use_rope = use_rope

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, relations_idx, labels_mask, sin_embeddings=None, cos_embeddings=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids)

        last_hidden_state = outputs.last_hidden_state
        # outputs: (batch_size, seq_len, linear_dim*2)
        outputs = self.linear(last_hidden_state)
        # query, key: (batch_size, ent_type_size, seq_len, linear_dim)
        query, key = torch.split(outputs, self.linear_dim, dim=-1)

        if self.use_rope:
            qw2 = torch.stack([-query[..., 1::2], query[..., ::2]], -1)
            qw2 = qw2.reshape(query.shape)
            query = query * cos_embeddings + qw2 * sin_embeddings

            kw2 = torch.stack([-key[..., 1::2], key[..., ::2]], -1)
            kw2 = kw2.reshape(key.shape)
            key = key * cos_embeddings + kw2 * sin_embeddings

        # 取sub、obj的emb
        batch_idx, sub_start_idx_list, sub_end_idx_list, obj_start_idx_list, obj_end_idx_list = [], [], [], [], []
        for i, relation_idx in enumerate(relations_idx):
            batch_idx.append([i]*len(relation_idx))
            sub_start_idx_list.append([r_idx[0] for r_idx in relation_idx])
            sub_end_idx_list.append([r_idx[1] for r_idx in relation_idx])
            obj_start_idx_list.append([r_idx[2] for r_idx in relation_idx])
            obj_end_idx_list.append([r_idx[3] for r_idx in relation_idx])

        sub_start_query = query[batch_idx, sub_start_idx_list, :]
        sub_end_query = query[batch_idx, sub_end_idx_list, :]
        sub_query = torch.cat([sub_start_query, sub_end_query], dim=-1)

        obj_start_key = key[batch_idx, obj_start_idx_list, :]
        obj_end_key = key[batch_idx, obj_end_idx_list, :]
        obj_key = torch.cat([obj_start_key, obj_end_key], dim=-1)

        logits = (sub_query * obj_key).sum(dim=-1)

        logits += (1.0 - labels_mask) * -1e12
        logits /= self.linear_dim ** 0.5

        return logits

class RelationInference(object):
    def __init__(self, model_path):
        if isinstance(model_path, list):
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path[0])  # 加载分词器
            self.model = [CustomModel.from_pretrained(sub_model_path) for sub_model_path in model_path]
        else:
            self.model = CustomModel.from_pretrained(model_path)
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)  # 加载分词器

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def predict(self, sentence: dict, threshold=0.0):
        outputs = self.tokenizer(sentence['sentence_text'], add_special_tokens=False, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = outputs.input_ids, outputs.attention_mask, outputs.token_type_ids

        position_ids = sentence['position_ids']
        position_ids = torch.tensor(position_ids, dtype=torch.long).unsqueeze(dim=0)

        relation_idx = sentence['relation_idx']

        pred_relation_idx = []

        input_ids, attention_mask, token_type_ids, position_ids = \
            input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device), position_ids.to(self.device),

        if isinstance(self.model, list):
            logits_list = []
            for model in self.model:
                model = model.to(self.device)
                # sinusoidal_position_embedding
                indices = torch.arange(0, model.linear_dim // 2, dtype=torch.float, device=self.device)
                indices = torch.pow(10000, -2 * indices / model.linear_dim)
                position_embeddings = position_ids[..., None] * indices[None, None, :]

                sin_embeddings = torch.sin(position_embeddings).repeat_interleave(2, dim=-1)  # sin_embeddings:(1,seg_len,linear_dim)
                cos_embeddings = torch.cos(position_embeddings).repeat_interleave(2, dim=-1)  # cos_embeddings:(1,seg_len,linear_dim)

                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
                last_hidden_state = outputs.last_hidden_state
                outputs = model.linear(last_hidden_state)
                query, key = torch.split(outputs, model.linear_dim, dim=-1)

                qw2 = torch.stack([-query[..., 1::2], query[..., ::2]], -1)
                qw2 = qw2.reshape(query.shape)
                query = query * cos_embeddings + qw2 * sin_embeddings

                kw2 = torch.stack([-key[..., 1::2], key[..., ::2]], -1)
                kw2 = kw2.reshape(key.shape)
                key = key * cos_embeddings + kw2 * sin_embeddings

                # 取sub、obj的emb
                sub_start_idx_list = [r_idx[0] for r_idx in relation_idx]
                sub_end_idx_list = [r_idx[1] for r_idx in relation_idx]
                obj_start_idx_list = [r_idx[2] for r_idx in relation_idx]
                obj_end_idx_list = [r_idx[3] for r_idx in relation_idx]

                sub_start_query = query[:, sub_start_idx_list, :]
                sub_end_query = query[:, sub_end_idx_list, :]
                obj_start_key = key[:, obj_start_idx_list, :]
                obj_end_key = key[:, obj_end_idx_list, :]

                sub_query = torch.cat([sub_start_query, sub_end_query], dim=-1)
                obj_key = torch.cat([obj_start_key, obj_end_key], dim=-1)

                logits = (sub_query * obj_key).sum(dim=-1)

                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=0)
            logits = logits.mean(dim=0)

            for idx in torch.where(logits > threshold)[0]:
                pred_relation_idx.append(idx.item())

        else:
            # sinusoidal_position_embedding
            indices = torch.arange(0, self.model.linear_dim // 2, dtype=torch.float)
            indices = torch.pow(10000, -2 * indices / self.model.linear_dim)
            position_embeddings = position_ids[..., None] * indices[None, None, :]

            sin_embeddings = torch.sin(position_embeddings).repeat_interleave(2, dim=-1)  # sin_embeddings:(1,seg_len,linear_dim)
            cos_embeddings = torch.cos(position_embeddings).repeat_interleave(2, dim=-1)  # cos_embeddings:(1,seg_len,linear_dim)

            outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
            last_hidden_state = outputs.last_hidden_state
            outputs = self.model.linear(last_hidden_state)
            query, key = torch.split(outputs, self.model.linear_dim, dim=-1)

            qw2 = torch.stack([-query[..., 1::2], query[..., ::2]], -1)
            qw2 = qw2.reshape(query.shape)
            query = query * cos_embeddings + qw2 * sin_embeddings

            kw2 = torch.stack([-key[..., 1::2], key[..., ::2]], -1)
            kw2 = kw2.reshape(key.shape)
            key = key * cos_embeddings + kw2 * sin_embeddings

            # 取sub、obj的emb
            sub_start_idx_list = [r_idx[0] for r_idx in relation_idx]
            obj_start_idx_list = [r_idx[2] for r_idx in relation_idx]
            sub_start_query = query[:, sub_start_idx_list, :]
            obj_start_key = key[:, obj_start_idx_list, :]

            logits = (sub_start_query * obj_start_key).sum(dim=-1)

            for idx in torch.where(logits > threshold)[0]:
                pred_relation_idx.append(idx.item())

        return pred_relation_idx

    def batch_predict(self, sentences, threshold=0.0):
        pred_tags_list = []
        for sentence in tqdm(sentences):
            pred_tags = self.predict(sentence, threshold)
            pred_tags_list.append(pred_tags)

        return pred_tags_list

def make_ann_needtodeal(TXT_needtodeal_path,ANN_needtodeal_path):
    with open(TXT_needtodeal_path, "r", encoding='utf-8') as f:

        dic_T2entity = {}
        relations = f.readlines()
        entities = eval(relations[0])

        with codecs.open(ANN_needtodeal_path, "w", encoding="utf-8") as w:

            for j in range(len(entities['ners'])):

                w.write('T' + str(j + 1) + '\t' + str(entities['ners'][j][2]) + ' ' + str(
                    entities['ners'][j][0]) + ' ' + str(
                    entities['ners'][j][1]) + '\t' + str(entities['ners'][j][3]) + '\n')
                dic_T2entity.update({str(entities['ners'][j][0]): 'T' + str(j + 1)})

            for i in range(len(relations) - 2):
                relation_list = relations[i + 1].split("\t", 3)

                relation_list[0] = ast.literal_eval(relation_list[0])
                relation_list[1] = ast.literal_eval(relation_list[1])

                w.write('R' + str(i + 1) + '\t' + str(relation_list[0][2]) + str(
                    relation_list[1][2]).capitalize() + ' ' + 'Arg1:' +
                        dic_T2entity.get(str(relation_list[0][0])) + ' ' + 'Arg2:' + dic_T2entity.get(
                    str(relation_list[1][0])) + '\n')

def make_finalresult_ann(ANN_needtodeal_path, ANN_finalresult_path,r_original_TXT_path,w_merge_ab_ann_path,file):
    global shiti_dict
    global guanxi_dict
    global shiti2_list_dict
    global guanxi2_list_dict
    global t_num
    global r_num
    global former_txt_lenth

    if (file.rfind('a') != (len(file) - 5)) and (file.rfind('b') != (len(file) - 5)):
        with codecs.open(ANN_needtodeal_path, "r", encoding="utf-8") as f:
            content = f.read()
            with codecs.open(ANN_finalresult_path, "w", encoding="utf-8") as f:
                f.write(content)

    elif (file.rfind('a') == (len(file) - 5)) or (file.rfind('b') == (len(file) - 5)):
        if file.rfind('a') == (len(file) - 5):
            with codecs.open(r_original_TXT_path, "r", encoding="utf-8") as f:
                former_txt = f.read()
                former_txt = former_txt.replace("\r\n", "\n")
                former_txt = former_txt.rstrip("\n")
                former_txt_lenth = len(former_txt)
            t_num = 0
            r_num = 0
            shiti_dict = {}
            guanxi_dict = {}
        if file.rfind('b') == (len(file) - 5):
            shiti2_list_dict = {}
            guanxi2_list_dict = {}

        with codecs.open(ANN_needtodeal_path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip("\r\n")
            line = line.strip("\n")

            while line != "":
                line_arr = line.split()
                for i in range(300):
                    if line_arr[0]==('%s' % ("T" + str(i))):
                        t_id_list = line_arr[0].split('T')
                        t_key = int(t_id_list[1])
                        if file.rfind('a') == (len(file) - 5):
                            shiti_dict[t_key] = line
                            t_num += 1
                        if file.rfind('b') == (len(file) - 5):
                            shiti2_list_dict[t_key+t_num] = line_arr
                            shiti2_list_dict[t_key + t_num][2] = int(shiti2_list_dict[t_key + t_num][2]) + former_txt_lenth
                            shiti2_list_dict[t_key + t_num][3] = int(shiti2_list_dict[t_key + t_num][3]) + former_txt_lenth
                            shiti_dict[t_key+t_num] = 'T' + str(t_key+t_num) + '\t' + shiti2_list_dict[t_key + t_num][1] + ' ' + str(shiti2_list_dict[t_key + t_num][2]) + ' ' + str(shiti2_list_dict[t_key + t_num][3]) + '\t' + shiti2_list_dict[t_key + t_num][4]

                        line = f.readline()
                        line = line.strip("\r\n")
                        line = line.strip("\n")

                    if line_arr[0] == ('%s' % ("R" + str(i))):
                        r_id_list = line_arr[0].split('R')
                        r_key = int(r_id_list[1])

                        if file.rfind('a') == (len(file) - 5):
                            guanxi_dict[r_key] = line
                            r_num += 1
                        if file.rfind('b') == (len(file) - 5):
                            object_id_list = line_arr[2].split('T')
                            object_id = int(object_id_list[1])
                            subject_id_list = line_arr[3].split('T')
                            subject_id = int(subject_id_list[1])

                            guanxi2_list_dict[r_key+r_num] = line_arr
                            guanxi2_list_dict[r_key + r_num][2] = 'Arg1:' + str(object_id + t_num)
                            guanxi2_list_dict[r_key + r_num][3] = 'Arg2:' + str(subject_id + t_num)
                            guanxi_dict[r_key + r_num] = 'R' + str(r_key + r_num) + '\t' + guanxi2_list_dict[r_key + r_num][1] + ' ' + guanxi2_list_dict[r_key + r_num][2] + ' ' + guanxi2_list_dict[r_key + r_num][3]

                        line = f.readline()
                        line = line.strip("\r\n")
                        line = line.strip("\n")

        if file.rfind('b') == (len(file) - 5):
            with codecs.open(w_merge_ab_ann_path, "w", encoding="utf-8") as f:
                for item1 in shiti_dict:
                    f.write(shiti_dict[item1] + '\n')
                for item2 in guanxi_dict:
                    f.write(guanxi_dict[item2] + '\n')



if __name__ == '__main__':
    # 处理数据集
    tokenizer_fast = BertTokenizerFast.from_pretrained('./chinese-roberta-wwm-ext-large')  # 加载tokenizer_fast

    # 模型推理
    time_predict_re_start = time.time()
    model_path = ['relation_model_1', 'relation_model_2', 'relation_model_3', 'relation_model_4', 'relation_model_5']
    inference = RelationInference(model_path=model_path)

    RE_data_dir= 'data/RESULT-RE'
    NER_data_dir= 'data/RESULT-NER/newTXT'
    for file in os.listdir(NER_data_dir):
        if file.find(".") == -1:
            continue
        file_name = file[0:file.find(".")]
        r_nertxt_path = os.path.join(NER_data_dir, "%s.txt" % file_name)
        w_json_path = "%s/newJSON/%s.json" % (RE_data_dir, file_name)

        test_data_processing(tokenizer_fast, r_nertxt_path, w_json_path)  # 生成测试集

        with open(w_json_path, 'r', encoding='utf-8') as f:
            testB_data = json.load(f)

        TXT_needtodeal_path= "%s/TXT_needtodeal/%s.txt" % (RE_data_dir, file_name)
        with open(TXT_needtodeal_path, 'w', encoding='utf-8') as f:
            for sentence in tqdm(testB_data):
                relation = sentence['relation']
                pred_relation_idx = inference.predict(sentence)

                f.write(json.dumps({'sent': sentence['sent'], 'ners': sentence['ners']}, ensure_ascii=False))
                for pred_idx in pred_relation_idx:
                    f.write('\n')
                    f.write(relation[pred_idx]+'\t'+'属性')
                f.write('\n\n')

    for file in os.listdir(NER_data_dir):
        if file.find(".") == -1:
            continue
        file_name = file[0:file.find(".")]
        TXT_needtodeal_path = "%s/TXT_needtodeal/%s.txt" % (RE_data_dir, file_name)
        ANN_needtodeal_path = "%s/ANN_needtodeal/%s.ann" % (RE_data_dir, file_name)
        ANN_finalresult_path = "%s/ANN_finalresult/%s.ann" % (RE_data_dir, file_name)
        r_original_TXT_path = "%s/original_TXT/%s.txt" % (RE_data_dir, file_name)
        w_merge_ab_ann_name = file_name[0:-2]
        w_merge_ab_ann_path = "%s/ANN_finalresult/%s.ann" % (RE_data_dir, w_merge_ab_ann_name)

        make_ann_needtodeal(TXT_needtodeal_path,ANN_needtodeal_path)
        make_finalresult_ann(ANN_needtodeal_path, ANN_finalresult_path,r_original_TXT_path, w_merge_ab_ann_path, file)

    time_predict_re_end = time.time()
    totaltime_predict_re = time_predict_re_end - time_predict_re_start
    per_test_time_predict_re = totaltime_predict_re / 50
    print("totaltime_predict_re: %f s" % totaltime_predict_re)
    print("per_test_time_predict_re: %f s" % per_test_time_predict_re)

