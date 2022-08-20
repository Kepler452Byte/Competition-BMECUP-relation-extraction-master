import json
import logging
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertTokenizerFast
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(filename='log/relation.txt', format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
writer = SummaryWriter('re_loss/television')

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


class CustomDataset(Dataset):
    """自定义Dataset"""

    def __init__(self, sentences: list):
        self._sentences = sentences

    def __len__(self):
        return len(self._sentences)

    def __getitem__(self, index):
        sentence = self._sentences[index]

        return {'text': sentence['sentence_text'],
                'position_ids': sentence['position_ids'],
                'relation': sentence['relation'],
                'relation_label': sentence['relation_label'],
                'relation_idx': sentence['relation_idx']}


class NerMetrics(object):
    """计算相关指标，精确率、召回率"""

    def __init__(self, save_metrics_history=False):
        self.save_metrics_history = save_metrics_history  # 是否保留历史指标
        self.history = []  # 记录每个epoch_end计算的指标，方便checkpoint

        self._tp = 0.0
        self._p = 0.0
        self._t = 0.0

    def reset(self):
        """结果重置"""
        self._tp = 0.0
        self._p = 0.0
        self._t = 0.0

    def update(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()

        f1_score = 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

        self._tp += torch.sum(y_true * y_pred)
        self._p += torch.sum(y_pred)
        self._t += torch.sum(y_true)

        return f1_score

    def compute_epoch_end(self):
        precision, recall, f1_score = 0.0, 0.0, 0.0
        if self._p > 0.0:
            f1_score = 2 * self._tp / (self._p + self._t)
            precision = self._tp / self._p
            recall = self._tp / self._t

        if self.save_metrics_history:
            self.history.append(f1_score)

        self.reset()

        return precision, recall, f1_score


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


class ModelCheckpoint(Callback):
    def __init__(self, save_path='output', mode='max', patience=10):
        super(ModelCheckpoint, self).__init__()
        self.path = save_path
        self.mode = mode
        self.patience = patience
        self.check_patience = 0
        self.best_value = 0.0 if mode == 'max' else 1e6  # 记录验证集最优值

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        """
        验证集计算结束后检查
        :param trainer:
        :param pl_module:
        :return:
        """
        if self.mode == 'max' and pl_module.valid_metrics.history[-1] >= self.best_value:
            self.check_patience = 0
            self.best_value = pl_module.valid_metrics.history[-1]
            logger.info(f'save best model with metric: {self.best_value:.5f}')
            pl_module.tokenizer.save_pretrained(self.path)
            pl_module.model.save_pretrained(self.path)  # 保存模型

        if self.mode == 'max' and pl_module.valid_metrics.history[-1] < self.best_value:
            self.check_patience += 1

        if self.mode == 'min' and pl_module.valid_metrics.history[-1] <= self.best_value:
            self.check_patience = 0
            self.best_value = pl_module.valid_metrics.history[-1]
            logger.info(f'save best model with metric: {self.best_value:.5f}')
            pl_module.tokenizer.save_pretrained(self.path)
            pl_module.model.save_pretrained(self.path)  # 保存模型

        if self.mode == 'min' and pl_module.valid_metrics.history[-1] > self.best_value:
            self.check_patience += 1

        if self.check_patience >= self.patience:
            trainer.should_stop = True  # 停止训练


class Relation(LightningModule):
    """采用pytorch-lightning训练的分类器"""

    def __init__(self, train_data: list, valid_data: list, model_path: str):
        super(Relation, self).__init__()

        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)  # 加载分词器
        self.model = CustomModel.from_pretrained(model_path, use_rope=True)  # 自定义的模型

        self.train_dataset = CustomDataset(sentences=train_data)  # 加载dataset
        self.valid_dataset = CustomDataset(sentences=valid_data)

        self.train_metrics = NerMetrics(save_metrics_history=False)  # 计算训练集的指标
        self.valid_metrics = NerMetrics(save_metrics_history=True)  # 计算验证集的指标

    @staticmethod
    def multilabel_cross_entropy(y_pred, y_true):
        """
        https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()

    def compute_loss(self, y_pred, y_true):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_pred = y_pred.reshape(batch_size*ent_type_size, -1)
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        loss = self.multilabel_cross_entropy(y_pred, y_true)

        return loss

    def train_collate_batch(self, batch):
        """
        处理训练集batch，主要是文本转成相应的tokens
        :param batch:
        :return:
        """
        relation_max_len = 0
        sentences = []
        for sentence in batch:
            sentences.append(sentence['text'])
            if len(sentence['relation_label']) >= relation_max_len:
                relation_max_len = len(sentence['relation_label'])

        outputs = self.tokenizer(sentences, truncation=True, max_length=800, padding=True,
                                 add_special_tokens=False, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = outputs.input_ids, outputs.attention_mask, outputs.token_type_ids

        batch_size, max_len = input_ids.shape
        position_ids = []
        for sentence in batch:
            pos_ids = sentence['position_ids'].copy()
            pad = [0]*(max_len-len(pos_ids))
            position_ids.append(pos_ids+pad)
        position_ids = torch.tensor(position_ids, dtype=torch.long)

        labels, labels_mask, relations_idx = [], [], []
        for sentence in batch:
            r_label = []
            for label in sentence['relation_label']:
                r_label.append(int(label == '属性'))
            pad = [0] * (relation_max_len - len(r_label))
            labels.append(r_label+pad)
            labels_mask.append([1]*len(r_label)+pad)

            relation_idx = sentence['relation_idx'].copy()
            for _ in pad:
                relation_idx.append([0, 0, 0, 0])
            relations_idx.append(relation_idx)

        labels = torch.tensor(labels, dtype=torch.long)
        labels_mask = torch.tensor(labels_mask, dtype=torch.long)

        indices = torch.arange(0, self.model.linear_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / self.model.linear_dim)
        position_embeddings = position_ids[..., None] * indices[None, None, :]
        sin_embeddings = torch.sin(position_embeddings).repeat_interleave(2, dim=-1)  # sin_embeddings:(batch_size,seg_len,linear_dim)
        cos_embeddings = torch.cos(position_embeddings).repeat_interleave(2, dim=-1)  # cos_embeddings:(batch_size,seg_len,linear_dim)

        return input_ids, attention_mask, token_type_ids, position_ids, labels, labels_mask, sin_embeddings, cos_embeddings, relations_idx

    def val_collate_batch(self, batch):
        """
        :param batch:
        :return:
        """
        relation_max_len = 0
        sentences = []
        for sentence in batch:
            sentences.append(sentence['text'])
            if len(sentence['relation_label']) >= relation_max_len:
                relation_max_len = len(sentence['relation_label'])

        outputs = self.tokenizer(sentences, truncation=True, max_length=800, padding=True,
                                 add_special_tokens=False, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = outputs.input_ids, outputs.attention_mask, outputs.token_type_ids

        batch_size, max_len = input_ids.shape
        position_ids = []
        for sentence in batch:
            pos_ids = sentence['position_ids'].copy()
            pad = [0]*(max_len-len(pos_ids))
            position_ids.append(pos_ids+pad)
        position_ids = torch.tensor(position_ids, dtype=torch.long)

        labels, labels_mask, relations_idx = [], [], []
        for sentence in batch:
            r_label = []
            for label in sentence['relation_label']:
                r_label.append(int(label == '属性'))
            pad = [0] * (relation_max_len - len(r_label))
            labels.append(r_label+pad)
            labels_mask.append([1]*len(r_label)+pad)

            relation_idx = sentence['relation_idx'].copy()
            for _ in pad:
                relation_idx.append([0, 0, 0, 0])
            relations_idx.append(relation_idx)

        labels = torch.tensor(labels, dtype=torch.long)
        labels_mask = torch.tensor(labels_mask, dtype=torch.long)

        indices = torch.arange(0, self.model.linear_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / self.model.linear_dim)
        position_embeddings = position_ids[..., None] * indices[None, None, :]
        sin_embeddings = torch.sin(position_embeddings).repeat_interleave(2, dim=-1)  # sin_embeddings:(batch_size,seg_len,linear_dim)
        cos_embeddings = torch.cos(position_embeddings).repeat_interleave(2, dim=-1)  # cos_embeddings:(batch_size,seg_len,linear_dim)

        return input_ids, attention_mask, token_type_ids, position_ids, labels, labels_mask, sin_embeddings, cos_embeddings, relations_idx

    def train_dataloader(self, train_batch_size=4):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate_batch
        )

    def val_dataloader(self, valid_batch_size=32):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.val_collate_batch,
        )

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, position_ids, labels, labels_mask, sin_embeddings, cos_embeddings, relations_idx = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, position_ids, relations_idx, labels_mask, sin_embeddings, cos_embeddings)

        loss = self.multilabel_cross_entropy(logits, labels)

        f_score = self.train_metrics.update(y_pred=logits, y_true=labels)

        self.print(f'epoch: {self.current_epoch}, global_step: {self.global_step},'
                   f' train_step_loss: {loss:.5f}, train_step_f_score: {f_score:.5f}')

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss = 0.0
        for output in outputs:
            loss += output['loss'].item()
        loss /= len(outputs)

        precision, recall, f_score = self.train_metrics.compute_epoch_end()

        self.print(
            f'epoch: {self.current_epoch}, global_step: {self.global_step}, train_loss: {loss:.5f}, train_f_score: {f_score:.5f}')
        logger.info(
            f'epoch: {self.current_epoch}, global_step: {self.global_step}, train_loss: {loss:.5f}, train_f_score: {f_score:.5f}')
        writer.add_scalar('train_re_loss', loss, self.global_step)
        writer.add_scalar('train_re_f_score', f_score, self.global_step)

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, position_ids, labels, labels_mask, sin_embeddings, cos_embeddings, relations_idx = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, position_ids, relations_idx, labels_mask, sin_embeddings, cos_embeddings)

        loss = self.multilabel_cross_entropy(logits, labels)

        f_score = self.valid_metrics.update(y_pred=logits, y_true=labels)

        self.print(f'epoch: {self.current_epoch}, global_step: {self.global_step}, '
                   f'valid_step_loss: {loss:.5f}, valid_step_f_score: {f_score:.5f}')

        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        loss = 0.0
        for output in outputs:
            loss += output['loss'].item()
        loss /= len(outputs)

        precision, recall, f_score = self.valid_metrics.compute_epoch_end()

        self.print(
            f'epoch: {self.current_epoch}, global_step: {self.global_step}, valid_loss: {loss:.5f}, valid_f_score: {f_score:.5f}')
        logger.info(
            f'epoch: {self.current_epoch}, global_step: {self.global_step}, valid_loss: {loss:.5f}, valid_f_score: {f_score:.5f}')
        writer.add_scalar('val_loss', loss, self.global_step)
        writer.add_scalar('valid_f_score', f_score, self.global_step)

    def configure_optimizers(self, bert_lr=2e-5, linear_lr=5e-5, weight_decay=0.01, total_step=49 * 50):
        """设置优化器"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if (n.startswith('bert') and not any(nd in n for nd in no_decay))],
             'weight_decay': weight_decay, 'lr': bert_lr},
            {'params': [p for n, p in self.model.named_parameters() if (n.startswith('bert') and any(nd in n for nd in no_decay))],
             'weight_decay': 0.0, 'lr': bert_lr},

            {'params': [p for n, p in self.model.named_parameters() if (n.startswith('linear') and not any(nd in n for nd in no_decay))],
             'weight_decay': weight_decay, 'lr': linear_lr},
            {'params': [p for n, p in self.model.named_parameters() if (n.startswith('linear') and any(nd in n for nd in no_decay))],
             'weight_decay': 0.0, 'lr': linear_lr}
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.2 * total_step), num_training_steps=total_step)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]


if __name__ == '__main__':
    # 处理数据集
    tokenizer_fast = BertTokenizerFast.from_pretrained('./chinese-roberta-wwm-ext-large')  # 加载tokenizer_fast
    train_raw_path = 'data/train.conll_convert.conll'  # train数据集
    train_path = 'data_relation/train.json'  # 生成train数据集
    train_data_processing(tokenizer_fast, train_raw_path, train_path)  # 生成训练集

    training = True
    if training:
        pl.seed_everything(1234)

        with open(train_path, 'r', encoding='utf-8') as f:
            train = json.load(f)

        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for n, (train_idx, valid_idx) in enumerate(kf.split(train)):
            logger.info(f'运行第{n+1}折......')

            train_data = [train[i] for i in train_idx]
            valid_data = [train[i] for i in valid_idx]

            checkpoint_callback = ModelCheckpoint(save_path=f'./relation_model_{n+1}', mode='max', patience=8)
            trainer = pl.Trainer(
                gpus=-1,
                precision=16,
                max_epochs=60,
                val_check_interval=1.0,
                callbacks=[checkpoint_callback],
                checkpoint_callback=False,
                logger=False,
                gradient_clip_val=0.0,
                distributed_backend=None,
                num_sanity_val_steps=-1,
                accumulate_grad_batches=8,
                check_val_every_n_epoch=1,
                progress_bar_refresh_rate=0,
            )
            ner = Relation(train_data=train_data, valid_data=valid_data, model_path='./chinese-roberta-wwm-ext-large')
            trainer.fit(ner)
            # break

    # 模型推理
    model_path = ['relation_model_1', 'relation_model_2', 'relation_model_3', 'relation_model_4', 'relation_model_5']
    inference = RelationInference(model_path=model_path)

    test_ner_path = 'data/test_ner.txt'  # 上一步得到的ner的结果
    test_path = 'data_relation/test.json'  # test关系数据
    submit_path = 'data_relation/submit.txt'  # 生成test结果
    test_data_processing(tokenizer_fast, test_ner_path, test_path)  # 生成测试集

    with open(test_path, 'r', encoding='utf-8') as f:
        testB_data = json.load(f)

    with open(submit_path, 'w', encoding='utf-8') as f:
        for sentence in tqdm(testB_data):
            relation = sentence['relation']
            pred_relation_idx = inference.predict(sentence)

            f.write(json.dumps({'sent': sentence['sent'], 'ners': sentence['ners']}, ensure_ascii=False))
            for pred_idx in pred_relation_idx:
                f.write('\n')
                f.write(relation[pred_idx]+'\t'+'属性')
            f.write('\n\n')
