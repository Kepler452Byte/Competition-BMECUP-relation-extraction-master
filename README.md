# 2022年生物医学创新设计竞赛
# 国家二等奖

## 参考项目:https://github.com/yeqingzhao/relation-extraction/blob/master/README.md

## 已经训练好的模型,可以直接进行预测
    链接：https://pan.baidu.com/s/1nE_abL-n3mA_dxCdOWrjig?pwd=6666 
    提取码：6666 
    --来自百度网盘超级会员V5的分享

## 运行环境

1. python==3.7.11
2. torch==1.9.0
3. transformers==4.11.3
4. pytorch-lightning==1.4.7
5. tqdm==4.62.3
6. numpy==1.21.0
7. scikit-learn==0.24.2

## 复现流程

1. 原始数据放在 `data`文件夹；
2. 下载 `chinese-roberta-wwm-ext-large`模型，已发送到邮箱，与开源的有些许差别，`vocab.txt`添加了一些专业的词汇，替换了 `[unused1]-[unused36]`；
3. 需要GPU(v100 32G)环境，训练ner模型，大概需要3小时。本地文件夹下运行 `global_pointer.py`，将生成数据 `data/labels.json`、`data/train.json`、`data/testB_ner.txt`、

   五折交叉的模型 `global_pointer_model_1`、`global_pointer_model_2`、`global_pointer_model_3`、`global_pointer_model_4`、`global_pointer_model_5`；
4. 需要GPU(v100 32G)环境，训练relation模型，大概需要12小时。本地文件夹下运行 `relation.py`，将生成数据 `data_relation/train.json`、`data_relation/submit_B.txt`、

   五折交叉的模型 `relation_model_1`、`relation_model_2`、`relation_model_3`、`relation_model_4`、`relation_model_5`；
5. 第4步生成的 `data_relation/submit_B.txt`为最终的结果；

## 服务类型接口使用

cuda版本    cuda:10.2 cudnn:10.2

*输入*:original dataset

    主 诉 :$ 胸 闷 气 促 伴 心 悸 2 天 $ ￣ 现 病 史 : $ 患 者 2 天 前 无 明 显 诱 因 下 出 现 胸 闷 气 促 ， 伴 心 悸 ， 呈 阵 发 性 ， 伴 大 汗 淋 漓 ， 无 胸 骨 后 压 榨 感 ， 伴 乏 力 ， 无 夜 间 阵 发 性 呼 吸 困 难 ， 无 畏 寒 发 热 ， ￣ 无 恶 心 呕 吐 ， 无 反 酸 烧 心 ， 无 头 痛 头 晕 ， 无 黑 曚 晕 厥 ， 无 咳 嗽 咳 痰 等 不 适 。 未 予 治 疗 ， 此 后 上 述 症 状 反 复 存 在 ， 遂 至 院 门 诊 就 诊 ， 为 进 一 步 治 疗 ， ￣ 门 诊 拟 ” 心 房 颤 动 ” 收 住 入 院 。

*输出*:官方要求的ANN文件

    T1  clinicalFeature 4 6 胸闷

    T2  clinicalFeature 6 8 气促

    T3  clinicalFeature 9 11    心悸

    T4  time 11 13  2天

    T5  time 22 25  2天前

    T6  cause 25 30 无明显诱因

    T7  clinicalFeature 33 35   胸闷

    T8  clinicalFeature 35 37   气促

    T9  clinicalFeature 39 41   心悸

    T10 modification 43 46  阵发性

    T11 clinicalFeature 48 52   大汗淋漓

在模型训练完成之后,项目中会生成 `global_pointer_model_1` ~`global_pointer_model_5` ;`relation_model_1` ~`relation_model_5` 这些文件,我们可以利用这些生成的文件,对输入的文本进行命名实体识别与关系抽取:

1. 首先运行 `predict-test-ner.py`  文件,得到各个文本的命名实体的预测结果
2. 然后运行 `predict-test-re.py` 文件,在 `relation-extraction-master\data\RESULT-RE\ANN_finalresult` 路径下生成与官方数据格式相同的ann文件
