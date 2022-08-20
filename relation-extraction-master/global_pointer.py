import itertools
import json
import logging
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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

logging.basicConfig(filename='log/global_pointer.txt', format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
writer = SummaryWriter('ner_loss/television')

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
                    start -= (i - 1)
                    end -= (i - 1)
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


class CustomDataset(Dataset):
    """自定义Dataset"""

    def __init__(self, sentences: list):
        self._sentences = sentences

    def __len__(self):
        return len(self._sentences)

    def __getitem__(self, index):
        sentence = self._sentences[index]

        return {'text': sentence['sent'],
                'tags': sentence['ners'],
                'spans': sentence['spans']}


class CustomMetrics(object):
    """计算相关指标，精确率、召回率"""

    def __init__(self, beta=1, save_metrics_history=False):
        self.beta = beta
        self.save_metrics_history = save_metrics_history  # 是否保留历史指标
        self.history = []  # 记录每个epoch_end计算的指标，方便checkpoint

        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return self.get_tp(class_name) / (self.get_tp(class_name) + self.get_fp(class_name))
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return self.get_tp(class_name) / (self.get_tp(class_name) + self.get_fn(class_name))
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return (1 + self.beta * self.beta) * (self.precision(class_name) * self.recall(class_name)) / \
                   (self.precision(class_name) * self.beta * self.beta + self.recall(class_name))
        return 0.0

    def accuracy(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name) + self.get_tn(class_name) > 0:
            return (self.get_tp(class_name) + self.get_tn(class_name)) / \
                   (self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name) + self.get_tn(
                       class_name))
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [self.accuracy(class_name) for class_name in self.get_classes()]
        if len(class_accuracy) > 0:
            return sum(class_accuracy) / len(class_accuracy)
        return 0.0

    def get_classes(self):
        all_classes = set(itertools.chain(
            *[list(keys) for keys in [self._tps.keys(), self._fps.keys(), self._tns.keys(), self._fns.keys()]]))
        all_classes = [class_name for class_name in all_classes if class_name is not None]
        all_classes.sort()
        return all_classes

    def reset(self):
        """结果重置"""
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def update(self, batch_true_tags: list, batch_pred_tags: list):
        for true_sentence_tags, pred_sentence_tags in zip(batch_true_tags, batch_pred_tags):
            # check for true positives, false positives and false negatives
            for (start, end, tag, pred) in pred_sentence_tags:
                if [start, end, tag, pred] in true_sentence_tags:
                    self.add_tp(tag)
                else:
                    self.add_fp(tag)

            for start, end, tag, pred in true_sentence_tags:
                if [start, end, tag, pred] not in pred_sentence_tags:
                    self.add_fn(tag)

        precision = self.precision()
        recall = self.recall()
        f_score = self.f_score()

        return precision, recall, f_score

    def compute_epoch_end(self):
        precision = self.precision()
        recall = self.recall()
        f_score = self.f_score()
        if self.save_metrics_history:
            self.history.append(f_score)
        self.reset()

        return precision, recall, f_score


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

class GlobalPointer(LightningModule):
    """采用pytorch-lightning训练的分类器"""

    def __init__(self, train_data: list, valid_data: list, labels_path: str, model_path: str):
        super(GlobalPointer, self).__init__()

        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        self.ent2id = labels['ent2id']
        self.id2ent = labels['id2ent']
        self.ent_type_size = len(self.ent2id)

        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)  # 加载分词器
        self.model = CustomModel.from_pretrained(model_path, ent_type_size=self.ent_type_size, use_rope=True,
                                                 attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3)  # 自定义的模型

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
        y_true = y_true.reshape(batch_size*ent_type_size, -1)
        loss = self.multilabel_cross_entropy(y_pred, y_true)

        return loss

    def train_collate_batch(self, batch):
        """
        处理训练集batch，主要是文本转成相应的tokens
        :param batch:
        :return:
        """
        sentences, spans_list = [], []
        for sentence in batch:
            sentences.append(sentence['text'])
            spans_list.append(sentence['spans'])

        outputs = self.tokenizer(sentences, truncation=True, max_length=512, padding=True, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = outputs.input_ids, outputs.attention_mask, outputs.token_type_ids

        batch_size, max_seq_len = input_ids.shape
        labels = np.zeros((batch_size, self.ent_type_size, max_seq_len, max_seq_len))
        for i, spans in enumerate(spans_list):
            for start, end, ent_type, ent_text in spans:
                labels[i, self.ent2id[ent_type], start, end] = 1

        labels = torch.tensor(labels, dtype=torch.long)

        # sinusoidal_position_embedding
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, self.model.linear_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / self.model.linear_dim)
        position_embeddings = position_ids * indices
        sin_embeddings = torch.sin(position_embeddings).repeat_interleave(2, dim=-1)
        cos_embeddings = torch.cos(position_embeddings).repeat_interleave(2, dim=-1)

        sin_embeddings = sin_embeddings[None, None, :, :]  # sin_embeddings:(1,1,seg_len,linear_dim)
        cos_embeddings = cos_embeddings[None, None, :, :]  # cos_embeddings:(1,1,seg_len,linear_dim)

        return input_ids, attention_mask, token_type_ids, labels, sin_embeddings, cos_embeddings

    def val_collate_batch(self, batch):
        """
        :param batch:
        :return:
        """
        sentences, spans_list = [], []
        for sentence in batch:
            sentences.append(sentence['text'])
            spans_list.append(sentence['spans'])

        outputs = self.tokenizer(sentences, truncation=True, max_length=512, padding=True, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = outputs.input_ids, outputs.attention_mask, outputs.token_type_ids

        batch_size, max_seq_len = input_ids.shape
        labels = np.zeros((batch_size, self.ent_type_size, max_seq_len, max_seq_len))
        for i, spans in enumerate(spans_list):
            for start, end, ent_type, ent_text in spans:
                labels[i, self.ent2id[ent_type], start, end] = 1

        labels = torch.tensor(labels, dtype=torch.long)

        # sinusoidal_position_embedding
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, self.model.linear_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / self.model.linear_dim)
        position_embeddings = position_ids * indices
        sin_embeddings = torch.sin(position_embeddings).repeat_interleave(2, dim=-1)
        cos_embeddings = torch.cos(position_embeddings).repeat_interleave(2, dim=-1)

        sin_embeddings = sin_embeddings[None, None, :, :]  # sin_embeddings:(1,1,seg_len,linear_dim)
        cos_embeddings = cos_embeddings[None, None, :, :]  # cos_embeddings:(1,1,seg_len,linear_dim)

        return input_ids, attention_mask, token_type_ids, labels, sin_embeddings, cos_embeddings

    def train_dataloader(self, train_batch_size=2):
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
        input_ids, attention_mask, token_type_ids, labels, sin_embeddings, cos_embeddings = batch

        logits = self.model(input_ids, attention_mask, token_type_ids, sin_embeddings, cos_embeddings)
        loss = self.compute_loss(logits, labels)

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
        writer.add_scalar('train_loss', loss, self.global_step)
        writer.add_scalar('train_f_score', f_score, self.global_step)

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels, sin_embeddings, cos_embeddings = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, sin_embeddings, cos_embeddings)

        loss = self.compute_loss(logits, labels)

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

    def configure_optimizers(self, bert_lr=2e-5, linear_lr=5e-5, weight_decay=0.01, total_step=98 * 60):
        # """设置优化器"""
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

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_step), num_training_steps=total_step)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]


if __name__ == '__main__':
    # 构造训练集数据
    tokenizer_fast = BertTokenizerFast.from_pretrained('./chinese-roberta-wwm-ext-large')  # 加载tokenizer_fast
    train_raw_path = 'data/train.conll_convert.conll'  # train数据集
    train_path = 'data/train.json'  # 生成train数据集
    labels_path = 'data/labels.json'  # 实体类型，编号
    data_processing(tokenizer_fast, train_raw_path, train_path, labels_path)

    # 模型训练
    training = True
    if training:
        pl.seed_everything(42)

        with open(train_path, 'r', encoding='utf-8') as f:
            train = json.load(f)

        oof_test = [0 for _ in train]
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for n, (train_idx, valid_idx) in enumerate(kf.split(train)):
            logger.info(f'运行第{n + 1}折......')

            train_data = [train[i] for i in train_idx]
            valid_data = [train[i] for i in valid_idx]

            checkpoint_callback = ModelCheckpoint(save_path=f'./global_pointer_model_{n+1}', mode='max', patience=10)
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
                accumulate_grad_batches=1,
                check_val_every_n_epoch=1,
                progress_bar_refresh_rate=0,
            )
            ner = GlobalPointer(train_data=train_data, valid_data=valid_data,
                                labels_path=labels_path, model_path='chinese-roberta-wwm-ext-large')
            trainer.fit(ner)

    # 模型推理
    model_path = ['global_pointer_model_1', 'global_pointer_model_2', 'global_pointer_model_3',
                  'global_pointer_model_4', 'global_pointer_model_5']  # 五折交叉验证模型
    inference = GlobalPointerInference(model_path=model_path, labels_path=labels_path)

    test_raw_path = 'data/test.conll_sent.conll'  # test数据集
    test_path = 'data/test_ner.txt'  # 生成test数据ner部分

    with open(test_raw_path, 'r', encoding='utf-8') as f:
        test_sentences = f.readlines()
    with open(test_path, 'w', encoding='utf-8') as f:
        for sentence in tqdm(test_sentences):
            sentence_pred = dict()
            sentence = sentence.strip()

            pred_tags = inference.predict(sentence)

            sentence_pred['sent'] = sentence
            sentence_pred['ners'] = pred_tags

            f.write(json.dumps(sentence_pred, ensure_ascii=False))
            f.write('\n\n')
