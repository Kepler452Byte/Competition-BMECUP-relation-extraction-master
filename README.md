# 2022������ҽѧ������ƾ���

## �ο���Ŀ:https://github.com/yeqingzhao/relation-extraction/blob/master/README.md

## �Ѿ�ѵ���õ�ģ��,����ֱ�ӽ���Ԥ��
    ���ӣ�https://pan.baidu.com/s/1nE_abL-n3mA_dxCdOWrjig?pwd=6666 
    ��ȡ�룺6666 
    --���԰ٶ����̳�����ԱV5�ķ���

## ���л���

1. python==3.7.11
2. torch==1.9.0
3. transformers==4.11.3
4. pytorch-lightning==1.4.7
5. tqdm==4.62.3
6. numpy==1.21.0
7. scikit-learn==0.24.2

## ��������

1. ԭʼ���ݷ��� `data`�ļ��У�
2. ���� `chinese-roberta-wwm-ext-large`ģ�ͣ��ѷ��͵����䣬�뿪Դ����Щ����`vocab.txt`�����һЩרҵ�Ĵʻ㣬�滻�� `[unused1]-[unused36]`��
3. ��ҪGPU(v100 32G)������ѵ��nerģ�ͣ������Ҫ3Сʱ�������ļ��������� `global_pointer.py`������������ `data/labels.json`��`data/train.json`��`data/testB_ner.txt`��

   ���۽����ģ�� `global_pointer_model_1`��`global_pointer_model_2`��`global_pointer_model_3`��`global_pointer_model_4`��`global_pointer_model_5`��
4. ��ҪGPU(v100 32G)������ѵ��relationģ�ͣ������Ҫ12Сʱ�������ļ��������� `relation.py`������������ `data_relation/train.json`��`data_relation/submit_B.txt`��

   ���۽����ģ�� `relation_model_1`��`relation_model_2`��`relation_model_3`��`relation_model_4`��`relation_model_5`��
5. ��4�����ɵ� `data_relation/submit_B.txt`Ϊ���յĽ����

## �������ͽӿ�ʹ��

cuda�汾    cuda:10.2 cudnn:10.2

*����*:original dataset

    �� �� :$ �� �� �� �� �� �� �� 2 �� $ �� �� �� ʷ : $ �� �� 2 �� ǰ �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ѹ ե �� �� �� �� �� �� �� ҹ �� �� �� �� �� �� �� �� �� �� η �� �� �� �� �� �� �� �� Ż �� �� �� �� �� �� �� �� �� ͷ ʹ ͷ �� �� �� �� �� �� �� �� �� �� �� �� ̵ �� �� �� �� δ �� �� �� �� �� �� �� �� ֢ ״ �� �� �� �� �� �� �� Ժ �� �� �� �� �� Ϊ �� һ �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ס �� Ժ ��

*���*:�ٷ�Ҫ���ANN�ļ�

    T1  clinicalFeature 4 6 ����

    T2  clinicalFeature 6 8 ����

    T3  clinicalFeature 9 11    �ļ�

    T4  time 11 13  2��

    T5  time 22 25  2��ǰ

    T6  cause 25 30 ����������

    T7  clinicalFeature 33 35   ����

    T8  clinicalFeature 35 37   ����

    T9  clinicalFeature 39 41   �ļ�

    T10 modification 43 46  ����

    T11 clinicalFeature 48 52   ������

��ģ��ѵ�����֮��,��Ŀ�л����� `global_pointer_model_1` ~`global_pointer_model_5` ;`relation_model_1` ~`relation_model_5` ��Щ�ļ�,���ǿ���������Щ���ɵ��ļ�,��������ı���������ʵ��ʶ�����ϵ��ȡ:

1. �������� `predict-test-ner.py`  �ļ�,�õ������ı�������ʵ���Ԥ����
2. Ȼ������ `predict-test-re.py` �ļ�,�� `relation-extraction-master\data\RESULT-RE\ANN_finalresult` ·����������ٷ����ݸ�ʽ��ͬ��ann�ļ�
