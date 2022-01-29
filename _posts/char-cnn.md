---
title: "Character-level convolutional networks for text classification"
layout: single
author: 
  - 제이엠
  - 꼬영
  - 제니
tags: 
  - CNN
  - NLP
date: January 3, 2022 11:39 PM
---

# Character-level convolutional networks for text classification

'''
property: NLP
작성일자: January 3, 2022 11:39 PM
팀원: 제이엠, 꼬영, 제니
'''

1. [**논문 선정 배경**](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
2. [Introduction](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
3. [Character-level Convolutional Networks](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
    3.1 [Key Modules](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
    3.2 [Character quantization](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
    3.3 [Model design](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
    3.4 [Data Augmentation using Thesaurus](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
4. [Comparison Models](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
    4.1 [Traditional Methods](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
    4.2 [Deep learning methods](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
5. [Large-scale Datasets and Results](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
    5.1 [Dataset](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
    5.2 [Result](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
    
6. [Discussion](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
7. [Conclusion and Outlook](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)
8. [Code](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232.md)

# 1. 논문 선정 배경

과거 텍스트 분류를 위해 CNN을 활용한 모델들은 입력값의 최소단위로 단어(embedded word vector)를 사용했으며, 보통 word2vec 임베딩된 단어 벡터들, TFIDF 정보, 혹은 n-gram 정보들을 취합한 bag of word이 주를 이루었습니다.

반면 본 논문은 기존 모델들의 접근방식이었던 단어보다 더 raw한 정보인 문자에 주목합니다. 이는 텍스트 분류를 위해 문자 단위를 ConvNet에 최초로 적용시켰다는 데 의미가 있습니다. 또한 문자를 사용함으로써 근본적인 언어 구조의 특징을 뽑아내고자한 점이 매우 인상깊었습니다. 따라서 어떠한 내용을 담고 있는지 자세히 살펴보고 함께 공유하고자 해당 논문을 선정하였습니다.

# 2. Introduction

텍스트 분류는 자연어 처리에 대한 고전적 주제입니다. 현재까지의 모든 텍스트 분류 기술들은 단어 수준에 관한 것이며, 그 중 몇몇 정렬된 단어 조합(예. n-grams)의 간단한 통계는 일반적으로 최고의 성능을 발휘합니다.

반면 많은 연구자들은 CNN이 음성 인식과 같은 raw signal로부터 정보를 추출하기 유용하다는 것을 발견했습니다. 본 논문에서는 문자 수준의 raw signal로 텍스트를 처리하기 위해 1D-CNN을 적용하는 법을 연구합니다. 또한 대규모의 데이터셋을 필요로 하는 CNN의 특성상 여러 데이터셋을 구축했습니다. 하지만 CNN은 단어에 대한 지식을(통사 또는 의미구조를 포함) 필요로 하지 않기 때문에 매우 용이합니다. 또한 이렇게 문자 기반으로 학습된 모델은 조금의 수정으로도 여러 언어에 적용될 수 있고, 철자 오류나 이모티콘도 자연스럽게 잘 학습시킬 수 있다는 장점이 있습니다.

# 3. Character-level Convolutional Networks

## 3.1 Key Modules

모델의 주된 구성은 단순히 1D Convolution만 계산하는 시간의 Conv. module입니다. 

이산 **input function** $g(x) \in [1,l] \rightarrow \mathbb{R}$ 와 이산 **kernel function** $f(x) \in [1,k] \rightarrow \mathbb{R}$을 가정합니다. 

다시 말하면, input function $g(x)$는 실수 공간 $[1,l]$ 내 원소로 정의되며, (kernel function) $f(x)$는 실수공간 $[1,k]$ 내 원소로 정의됩니다.

*stride $d$*를 갖는 $f(x)$와 $g(x)$의 **Convolution** $h(y) \in [1,[(l-k)/d]+1] \rightarrow \mathbb{R}$는 다음과 같이 정의됩니다.

- *Stride* : 입력데이터에 필터를 적용할 때 간격을 조절하는 것, 즉 필터가 이동할 간격의미.
    
    ex) Stride = 1인 합성곱
    
    [https://t1.daumcdn.net/cfile/tistory/992CA63F5C38B80029](https://t1.daumcdn.net/cfile/tistory/992CA63F5C38B80029)
    

# $h(y) = \sum_{x=1}^{k} f(x) \cdot g(y\cdot d-x+c)$

단, 이 때 $c=k-d+1$ 로, *오프셋 상수*입니다. 

- *오프셋 상수 : 동일 오브젝트 안에서 오브젝트 처음부터 주어진 요소나 지점까지의 변위차를 나타내는 정수형.*

Vision에서 전통적인 Convolution Net과 마찬가지로, 본 모듈은 input $g_{i}(x)$와 output $h_{j}(y)$의 집합에서 가중치(weights)라고 불리는 kernel function $f_{ij}(x)$ ($i = 1,2,...,m$  and $j=1,2,...n$)의 집합에 대해서 매개변수화 됩니다. 

$g_{i}$ : input feature

$h_{j}$: output feature

$m$: input feature size

$n$: output feature size

따라서, output $h_{j}(y)$는 $g_{i}(x)$와 $f_{ij}(x)$의 Convolution을 $i$에 대해 합하여 얻어집니다.

더 깊은 모델을 훈련시키는데 도움이 된 핵심 모듈 중 하나는 시간 max-pooling입니다. 컴퓨터 비전에서 사용되는 max-pooling의 1-D 버전이라고 생각하면 됩니다. (2차원 → 1차원으로 차원 축소)

input function $g(x) \in [1,l] \rightarrow \mathbb{R}$가 주어졌을 때, $g(x)$의 **max-pooling function** $h(y) \in [1,[(l-k)/d]+1] \rightarrow \mathbb{R}$은 다음과 같이 정의 됩니다.

$$
h(y) = \max_{x=1}^{k} g(y \cdot d -x +c)
$$

단, 이 때 $c=k-d+1$ 로, *오프셋 상수*입니다. 

바로 이 pooling module은 6개의 layer보다 더 깊은 ConvNets를 학습가능하게 만들었습니다.

모델의 비선형성은 thresholding function $h(x) = max \{0,x\}$ 이며, 이것은 Convolutional layer를 Rectified Linear Units(ReLUs)와 비슷하게 만듭니다. 

사용된 알고리즘은 미니배치 사이즈가 128인 확률적 경사하강법(SGD)이며, momentum 0.9, initial step size는 0.01을 사용하였습니다.

각 epoch는 클래스 전체에서 균일하게 샘플링 되어 고정된 수 만큼 무작위로 train sample을 취합니다.

이 모델은 인코딩된 문자 시퀀스를 입력으로 받아들입니다. 여기서 인코딩은 $m$개의 알파벳에 대해 one-hot 인코딩 방식을 사용했다. 따라서 각 입력은 고정 길이가 $l_0$ 인 $m$차원의 벡터가 되며, 전체 시퀀스는 $l_0 * m$차원의 행렬로 표현될 것입니다. 이때 길이가 $l_0$을 초과하는 모든 문자는 무시되며, 공백 문자를 포함하여 알파벳이 아닌 모든 문자는 모두 제로 벡터로 양자화됩니다.

![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled.png)

이 모델에서는 알파벳을 총 70개의 문자로 정의했습니다$(m=70)$ . 26개의 영어 문자, 10개의 숫자, 그리고 33개의 특수문자와 줄 내림 문자로 구성되었으며 소문자로 입력받도록 하였습니다. 전체 알파벳은 다음과 같습니다.

![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%201.png)

## 3.3 Model design

![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%202.png)

최종적으로 2개의 ConvNet을 설계했습니다. 하나는 많은 feature를 가지는 ConvNet이고, 다른 하나는 적은 feature를 가지는 ConvNet으로 feature 수를 제외하고는 모두 동일합니다. 이들은 6개의 Convolutional layer와 3개의 fully-connected layer로 이루어진 총 9개의 layer로 표현됩니다.

더욱 세부적으로 살펴보겠습니다. 입력의 feature 수는 70이며 입력 길이는 1014입니다. 이는 앞서 언급한 one-hot 인코딩을 사용했기 때문에 70차원의 벡터가 되는 것이며, 1014개의 문자까지만 입력으로 받는다는 것을 의미합니다. 본 논문에 따르면 이 정도 길이의 문자 시퀀스라면 텍스트의 대부분의 주요 내용을 잡아낼 수 있다고 합니다. 

정규화를 위하여 3개의 fully-connected layer 사이에 dropout을 2번 사용했으며, 확률은 0.5로 설정했습니다. 가중치 초기화는 가우시안 분포를 따르도록 하고 분포의 평균과 분산은 큰 모델에 대해서는 (0, 0.02)로 작은 모델은 (0, 0.05)로 설정했습니다.

![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%203.png)

![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%204.png)

위 표는 앞서 언급했듯이 feature 수에 따른 크고 작은 모델의 세부 구조를 보여줍니다. 큰 모델은 feature 수를 1024, 작은 모델은 256으로 설정하여 convolution을 진행했습니다. 즉 다른 크기의 필터를 사용했다고 이해할 수 있습니다. 참고로 stride를 1로 하고 Pooling과정에서 overlap되는 부분이 없게 하였습니다.  

## 3.4 Data Augmentation using Thesaurus

데이터 증강은 딥러닝 모델에서 일반화 정도를 향상시킬 수 있는 효과적인 방법입니다. 하지만 텍스트의 경우 문자의 순서가 매우 중요하기 때문에 이미지나 음성 인식에서처럼 데이터 변환을 통해 데이터를 늘리는 것은 바람직하지 않습니다. 사실 가장 좋은 방법은 사람이 직접 문장을 바꿔쓰는 것입니다. 하지만 이는 데이터의 크기가 증가할수록 비용이 많이 소요되므로 본 논문에서는 단어나 구를 유의어로 대체시키는 방식을 택했습니다(English Thesaurus 사용).

먼저 주어진 텍스트에서 대체 가능한 모든 단어를 추출합니다. 그런 다음 $P[r]$ ~ $p^r$를 통해 샘플링 된  $r$개의 단어를 유의어로 대체하였으며, 동일한 기하분포인 $P[r]$ ~ $q^s$로부터 샘플링된 s로부터 유의어의 index를 결정했습니다. 기하분포를 사용하였기 때문에 자주 사용되는 의미와 멀어질수록 유의어가 선택될 가능성이 적을 것이라고 추측할 수 있을 것입니다.

ex) [여아, 소녀, 처녀, 아줌마]일 때,
여아의 유의어로 선택될 확률: 소녀 > 처녀 > 아줌마

# 4. Comparison Models

Character CNN 모델을 전통적인 선형모델과 비 선형의 딥러닝모델로 비교한 결과에 대한 내용입니다.

## 4**.1 Traditional Method**(기존의 전통적인 선형모델) : 모두 다항 로지스틱 회귀분석 사용

- **Bag-of-words and its TFIDF** : bag-of-words 모델은 각 데이터셋에 빈도가 높은 50000개 단어들로 구성됨
    - Bag of Words :  단어들의 순서는 전혀 고려하지 않고, 단어들의 **출현 빈도(frequency)**에만 집중하는 텍스트 데이터의 수치화 표현 방법
    - TFIDF
        
        [TFIDF 공식]
        
        ![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%205.png)
        
        TF : 각 문서에서의 각 단어의 등장 빈도
        
        DF : 특정 단어 t가 등장한 문서의 수
        
        IDF : DF의 반비례
        
        1) 해당 문서에서 나타난 횟수가 많을수록(TF)
        
        2) 다른 문서에서 나타난 횟수가 적을수록(IDF) 
        
        → **해당 문서를 대표하는 키워드**
        
        → 그저 단어의 빈도만을 세는 bag-of-words와는 다름!
        
- **Bag-of-ngrams and its TFIDF :** 5-grams까지의 n-gram에서 가장 빈도가 높은 상위 500,000개로 구성됨, TFIDF는 동일한 과정
    - n-gram에 대한 이해
        
        ![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%206.png)
        
- **Bag-of-means on word embedding :** word2vec을 적용한 것에 대해 k-means을 사용함, 이를 통해 나온 단어를 클러스터링 된 단어들의 대표 단어로 사용함

      embedding 차원 : 300 ****

*Word2vec의 단어의 벡터화 : 주변 (context window)에 같은 단어가 나타나는 단어일수록 비슷한 벡터값을 갖음

## 4.2 Deep Learning Methods  : **모두 word2vec을 이용하여 단어를 임베딩함(embedding size : 300)**

- Word-based ConvNets
    
    *우리의 모델은 character 기반이고, 사용한 비교군 딥러닝 모델은 단어 기반임
    
- LSTM(Long-short term memory)
    
    학습 시 gradient clipping과 multinomial logistic regression을 사용하였음
    

# 5. Large-scale Datasets and Results

CNN은 보통 큰 데이터셋에 효과적인데 특히 우리의 모델처럼 character단위의 low-level의 raw features들에 더욱 필요합니다. 하지만 대부분의 텍스트 분류를 위한 데이터의 크기가 작으므로 필자는 데이터셋을 만들었습니다. 

## 5.1 Dataset

![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%207.png)

- AG’s news corpus
- Sogou news corpus
- DBPedia ontology dataset
- Yelp reviews
- Yahoo! Answers dataset
- Amazon reviews

## 5.2 Result

위의 데이터셋으로 모델들을 돌린 testing error(%)를 나타낸 표입니다. (값이 작을수록 좋은 것)

good : 파란색, bad : 빨간색에 해당

Lg : large

Sm : small

w2v : word2vec

LK : lookup table

Th : thesaurus

![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%208.png)

# 6. Discussion

1) Character level ConvNet은 효과적인 방법입니다.

    단어 말고도 character 단위로도 텍스트 분류에 효과적인 방법이 될 수 있음을 보여줍니다.

2) 데이터셋의 크기는 traditional한 모델과 ConvNets 모델들 사이에서 성능 차이를 보입니다.

    작은 데이터셋 → 전통적인 NLP 모델이 성능 우수

    큰 데이터셋 → ConvNets 모델이 성능 우수 

    ⇒ 학습을 위해 많은 데이터를 필요로하는 CNN의 특성 때문입니다.

3) ConvNet은 사용자가 만든 데이터에서 좋습니다 → real world에 더 적합한 데이터임을 의미합니다.

    (하지만 convnet이 정말 오타나 이모티콘의 문자들에 강한지는 실험이 더 필요한 상태라고 합니다.)

4) 알파벳의 선택에 따라 성능이 많이 달라집니다.

    대문자를 추가하여 모델링하였을 때 성능이 좋지 못했습니다.

    저자들은 대, 소문자간의 의미차이가 실제로 존재하지 않기 때문에, 소문자만 사용했을때 regularization effect를 가져온다고 분석하였습니다.

5) task에 따른 성능 차이가 없음을 보여줍니다.

    감성 분석과 토픽분류에 대한 두가지 task에 성능을 확인해본 결과, 별다른 차이가 없었습니다.

6) Word2Vec 기반의 k-means 클러스터링을 진행하여 임베딩하였을 때, 모든 데이터셋에 대해 성능이 좋지 못하였습니다(text classification task에서)

    분산 표현을 단순하게 활용하여 역효과가 생겼다고 합니다.

7) 모든 데이터셋에 있어 최적의 모델은 없다는 점입니다.

    결국, 실험을 통해 데이터셋에 가장 적합한 모델을 찾아야 합니다.

# 7. Conclusion and Outlook

이 논문은 character-level의 convolutional networks가 text classification에서 효과적으로 사용될 수 있음을 보여줍니다. 큰 데이터셋을 사용하면서 많은 전통적, 혹은 딥러닝 방법들과 character cnn 모델을 비교해보았을 때, 데이터셋의 크기 혹은 어떤 알파벳을 사용했는지 등의 많은 요인들로 모델의 결과가 달라지기도 한다는 점이 존재합니다.

+)

Word-based CNN과 더불어 본 논문은 Text Classification을 위한 Character level의 CNN 모델을 제안하고 있습니다. 현재는 본 논문의 방법이 많이 사용되지는 않지만, 문자 혹은 문장 단위가 아니라 character 단위로도 text classification이 가능하다는 점이 인상깊었던것 같습니다

# 8. Code

논문에 나왔던 AG’s News dataset을 이용해 학습을 진행하였습니다. 코드는 Colab으로 진행하였습니다.

[train.csv](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/train.csv)

[test.csv](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/test.csv)

0) Load Data

```python
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model

from google.colab import drive
drive.mount('/content/drive')

train_df=pd.read_csv('AG_news/train.csv')
test_df=pd.read_csv('AG_news/test.csv')

train_df.rename(columns={'Class Index':0,'Title':1,'Description':2},inplace=True)
test_df.rename(columns={'Class Index':0,'Title':1,'Description':2},inplace=True)

# concatenate column 1 and column 2 as one text
for df in [train_df, test_df]:
  df[1] = df[1] + df[2]
  df = df.drop([2], axis=1)
```

1) Preprocessing

- 텍스트 소문자 변경

```python
train_texts = train_df[1].values 
train_texts = [s.lower() for s in train_texts]

test_texts = test_df[1].values 
test_texts = [s.lower() for s in test_texts]
```

- Tokenizer

```python
# Initialization
tk=Tokenizer(num_words=None, char_level=True,oov_token='UNK')

# Fitting
tk.fit_on_texts(train_texts)
```

- Construct Vocab

```python
# construct a new vocabulary
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

# Use char_dict to replace the tk.word_index
tk.word_index = char_dict.copy()
# Add 'UNK' to the vocabulary
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
# oov_token: Out Of Vocabulary (oov) -> 모르는 단어로 인해 문제를 푸는 것이 까다로워지는 상황 처리함

# Convert string to index
train_sequences = tk.texts_to_sequences(train_texts)
test_texts = tk.texts_to_sequences(test_texts)
```

- Padding

```python
train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
test_data = pad_sequences(test_texts, maxlen=1014, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')
test_data = np.array(test_data, dtype='float32')
```

- Get Label

```python
train_classes = train_df[0].values
train_class_list = [x - 1 for x in train_classes]

test_classes = test_df[0].values
test_class_list = [x - 1 for x in test_classes]

from tensorflow.keras.utils import to_categorical
train_classes = to_categorical(train_class_list)
test_classes = to_categorical(test_class_list)
```

2) Char CNN

- Parameter

```python
input_size = 1014
vocab_size = len(tk.word_index)
embedding_size = 69
conv_layers = [[256, 7, 3],
               [256, 7, 3],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, 3]]

fully_connected_layers = [1024, 1024]
num_of_classes = 4
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'
```

- Embedding Layer

```python
# Embedding weights
embedding_weights = []  # (70, 69)
embedding_weights.append(np.zeros(vocab_size))  # (0, 69)

for char, i in tk.word_index.items():  # from index 1 to 69
    onehot = np.zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)
print('Load')

# Embedding layer Initialization
embedding_layer = Embedding(vocab_size + 1,
                            embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])
```

- Model Construction

```python
# Input
inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)
# Embedding           
x = embedding_layer(inputs)
# Conv
for filter_num, filter_size, pooling_size in conv_layers:
    x = Conv1D(filter_num, filter_size)(x)
    x = Activation('relu')(x)
    if pooling_size != -1:
        x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)
x = Flatten()(x)  # (None, 8704)
# Fully connected layers
for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
    x = Dropout(dropout_p)(x)
# Output Layer
predictions = Dense(num_of_classes, activation='softmax')(x)
# Build model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Adam, categorical_crossentropy
model.summary()
```

![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%209.png)

- Shuffle

```python
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)

x_train = train_data[indices]
y_train = train_classes[indices]

x_test = test_data
y_test = test_classes
```

- Training

```python
learning_history=model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=128,
          epochs=10,
          verbose=2)
```

- Result

```python
import matplotlib.pyplot as plt

hist = pd.DataFrame(learning_history.history)
hist['epoch'] = learning_history.epoch
hist.tail()

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(hist['epoch'], hist['accuracy'], label = 'Train accuracy')
plt.plot(hist['epoch'], hist['val_accuracy'], label = 'Val accuracy')
plt.legend()
plt.show()
```

![Untitled](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/Untitled%2010.png)

*참고자료

[https://reniew.github.io/29/](https://reniew.github.io/29/)

[https://supkoon.tistory.com/38](https://supkoon.tistory.com/38)

[https://arxiv.org/abs/1509.01626](https://arxiv.org/abs/1509.01626)

[1주차 인스타그램 포스트](Character-level%20convolutional%20networks%20for%20text%20cl%20722cf4cfd2004f57a33c6a1db5dc9232/1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A1%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%20%E1%84%91%E1%85%A9%E1%84%89%E1%85%B3%E1%84%90%E1%85%B3%201947abf833a2419baa2ad3529c791aca.md)
