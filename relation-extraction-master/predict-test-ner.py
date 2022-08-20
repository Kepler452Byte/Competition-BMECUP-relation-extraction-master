import json
import os
import torch
import time
from torch import nn
from tqdm import tqdm
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertTokenizerFast



def data_processing(tokenizer_fast,
                    train_raw_path='data_raw/train.conll_convert.conll',
                    train_path='data_raw/train.json',
                    labels_path='data_raw/labels.json'):
    ents = set()
    sentences = []
    with open(train_raw_path, 'r', encoding='utf-8') as f:
        for sentence in tqdm(f.readlines()):
            try:
                sentence = json.loads(sentence)
            except Exception as e:
                continue

            outputs = tokenizer_fast(sentence['sent'], return_offsets_mapping=True)

            # 处理原句空格
            offset_mapping = []
            for i, (start, end) in enumerate(outputs["offset_mapping"]):
                if (end > 0) and (i >= 2):
                    start -= (i-1)
                    end -= (i-1)
                offset_mapping.append((start, end))

            ent2token_spans = []
            for ent in sentence['ners']:
                ent_start, ent_end, ent_type, ent_text = ent[0], ent[1], ent[2], ent[3]
                ents.add(ent_type)  # 记录实体种类

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

                ent2token_spans.append([ent_start_token_idx, ent_end_token_idx, ent_type, ent_text])

            sentence['spans'] = ent2token_spans
            sentences.append(sentence)

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    ent2id = dict()
    id2ent = dict()
    for i, ent in enumerate(sorted(ents)):
        ent2id[ent] = i
        id2ent[i] = ent
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({'ent2id': ent2id, 'id2ent': id2ent}, f, ensure_ascii=False, indent=2)

class CustomModel(BertPreTrainedModel):
    """自定义模型"""

    def __init__(self, config, ent_type_size, use_rope=True):
        super(CustomModel, self).__init__(config)
        self.bert = BertModel(config)  # transformers的写法，方便保存，加载模型

        self.linear_dim = int(config.hidden_size / config.num_attention_heads)  # 默认
        self.linear = nn.Linear(config.hidden_size, ent_type_size * self.linear_dim * 2)

        self.ent_type_size = ent_type_size
        self.use_rope = use_rope

    def forward(self, input_ids, attention_mask, token_type_ids, sin_embeddings=None, cos_embeddings=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state

        batch_size, seq_len = input_ids.shape

        # outputs: (batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.linear(last_hidden_state)

        # outputs: (batch_size, seq_len, ent_type_size, linear_dim*2)
        outputs = outputs.view((batch_size, seq_len, self.ent_type_size, self.linear_dim * 2))
        outputs = outputs.permute(0, 2, 1, 3)
        # query, key: (batch_size, ent_type_size, seq_len, linear_dim)
        query, key = torch.split(outputs, self.linear_dim, dim=-1)

        if self.use_rope:
            qw2 = torch.stack([-query[..., 1::2], query[..., ::2]], -1)
            qw2 = qw2.reshape(query.shape)
            query = query * cos_embeddings + qw2 * sin_embeddings

            kw2 = torch.stack([-key[..., 1::2], key[..., ::2]], -1)
            kw2 = kw2.reshape(key.shape)
            key = key * cos_embeddings + kw2 * sin_embeddings

        logits = torch.matmul(query, key.transpose(-1, -2))  # logits: (batch_size, ent_type_size, seq_len, seq_len)

        # 构建mask
        extended_attention_mask = attention_mask[:, None, None, :] * torch.triu(torch.ones_like(logits))
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e12

        logits += extended_attention_mask
        logits /= self.linear_dim ** 0.5

        return logits

class GlobalPointerInference(object):
    def __init__(self, model_path, labels_path: str, use_rope=True):
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        self.ent2id = labels['ent2id']
        self.id2ent = labels['id2ent']
        self.ent_type_size = len(self.ent2id)

        if isinstance(model_path, list):
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path[0])  # 加载分词器
            self.model = [CustomModel.from_pretrained(sub_model_path, ent_type_size=self.ent_type_size, use_rope=True) for sub_model_path in model_path]
        else:
            self.model = CustomModel.from_pretrained(model_path, ent_type_size=self.ent_type_size, use_rope=True)
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)  # 加载分词器

        self.use_rope = use_rope

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def decode_ent(self, text, pred_matrix, threshold=0.0):
        outputs = self.tokenizer(text, return_offsets_mapping=True)
        text = text.replace(' ', '')  # 处理原句空格

        offset_mapping = []
        for i, (start, end) in enumerate(outputs["offset_mapping"]):
            if (end > 0) and (i >= 2):
                start -= (i - 1)
                end -= (i - 1)
            offset_mapping.append((start, end))

        ent_list = []
        for ent_type_id, token_start_idx, token_end_idx in zip(*torch.where(pred_matrix > threshold)):
            ent_type_id, token_start_idx, token_end_idx = ent_type_id.item(), token_start_idx.item(), token_end_idx.item()
            ent_type = self.id2ent[str(ent_type_id)]
            ent_char_span = [offset_mapping[token_start_idx][0], offset_mapping[token_end_idx][1]]
            ent_text = text[ent_char_span[0]:ent_char_span[1]]

            ent_list.append([ent_char_span[0], ent_char_span[1], ent_type, ent_text])

        return ent_list

    @torch.no_grad()
    def predict(self, sentence: str, threshold=0.0):
        outputs = self.tokenizer(sentence, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = outputs.input_ids, outputs.attention_mask, outputs.token_type_ids

        if isinstance(self.model, list):
            logits_list = []
            for model in self.model:
                model = model.to(self.device)
                if self.use_rope:
                    # sinusoidal_position_embedding
                    position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.float).unsqueeze(-1)
                    indices = torch.arange(0, model.linear_dim // 2, dtype=torch.float)
                    indices = torch.pow(10000, -2 * indices / model.linear_dim)
                    position_embeddings = position_ids * indices
                    sin_embeddings = torch.sin(position_embeddings).repeat_interleave(2, dim=-1)
                    cos_embeddings = torch.cos(position_embeddings).repeat_interleave(2, dim=-1)

                    sin_embeddings = sin_embeddings[None, None, :, :]  # sin_embeddings:(1,1,seg_len,linear_dim)
                    cos_embeddings = cos_embeddings[None, None, :, :]  # cos_embeddings:(1,1,seg_len,linear_dim)

                    input_ids, attention_mask, token_type_ids, sin_embeddings, cos_embeddings = \
                        input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device), sin_embeddings.to(self.device), cos_embeddings.to(self.device)
                    logits = model(input_ids, attention_mask, token_type_ids, sin_embeddings, cos_embeddings)
                else:
                    logits = model(input_ids, attention_mask, token_type_ids)
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=0)
            logits = logits.mean(dim=0)
        else:
            if self.use_rope:
                # sinusoidal_position_embedding
                position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.float).unsqueeze(-1)
                indices = torch.arange(0, self.model.linear_dim // 2, dtype=torch.float)
                indices = torch.pow(10000, -2 * indices / self.model.linear_dim)
                position_embeddings = position_ids * indices
                sin_embeddings = torch.sin(position_embeddings).repeat_interleave(2, dim=-1)
                cos_embeddings = torch.cos(position_embeddings).repeat_interleave(2, dim=-1)

                sin_embeddings = sin_embeddings[None, None, :, :]  # sin_embeddings:(1,1,seg_len,linear_dim)
                cos_embeddings = cos_embeddings[None, None, :, :]  # cos_embeddings:(1,1,seg_len,linear_dim)

                logits = self.model(input_ids, attention_mask, token_type_ids, sin_embeddings, cos_embeddings).squeeze()
            else:
                logits = self.model(input_ids, attention_mask, token_type_ids).squeeze()

        pred_tags = self.decode_ent(sentence, logits, threshold)
        pred_tags = sorted(pred_tags, key=lambda x: x[0])

        return pred_tags

    def batch_predict(self, sentences, threshold=0.0):
        pred_tags_list = []
        for sentence in tqdm(sentences):
            pred_tags = self.predict(sentence, threshold)
            pred_tags_list.append(pred_tags)

        return pred_tags_list


if __name__ == '__main__':
    # 构造训练集数据
    tokenizer_fast = BertTokenizerFast.from_pretrained('./chinese-roberta-wwm-ext-large')  # 加载tokenizer_fast
    train_raw_path = 'data/train.conll_convert.conll'  # train数据集
    train_path = 'data/train.json'  # 生成train数据集
    labels_path = 'data/labels.json'  # 实体类型，编号
    data_processing(tokenizer_fast, train_raw_path, train_path, labels_path)

    # 模型推理
    time_predict_ner_start = time.time()
    model_path = ['global_pointer_model_1', 'global_pointer_model_2', 'global_pointer_model_3',
                  'global_pointer_model_4', 'global_pointer_model_5']  # 五折交叉验证模型
    inference = GlobalPointerInference(model_path=model_path, labels_path=labels_path)

    data_dir = 'data/RESULT-NER\\'
    for file in os.listdir(data_dir):
        if file.find(".") == -1:
            continue
        file_name = file[0:file.find(".")]
        # r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.conll" % file_name)
        # 设置存放新文件夹的路径
        w_path = "%s/newTXT/%s.txt" % (data_dir, file_name)
        with open(r_txt_path, 'r', encoding='utf-8') as f:
            testB_sentences = f.readlines()
        with open(w_path, 'w', encoding='utf-8') as f:
            for sentence in tqdm(testB_sentences):
                sentence_pred = dict()
                sentence = sentence.strip()
                pred_tags = inference.predict(sentence)
                sentence_pred['sent'] = sentence
                sentence_pred['ners'] = pred_tags

                f.write(json.dumps(sentence_pred, ensure_ascii=False))
                f.write('\n\n')

    time_predict_ner_end = time.time()
    totaltime_predict_ner = time_predict_ner_end - time_predict_ner_start
    per_test_time_predict_ner = totaltime_predict_ner / 50
    print("totaltime_predict_ner: %f" % totaltime_predict_ner)
    print("per_test_time_predict_ner: %f" % per_test_time_predict_ner)