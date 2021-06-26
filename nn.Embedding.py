''' nn.Embedding()
1. 임베딩 층을 만들어 훈련 데이터로부터 임베딩 벡터를 학습하는 방법
2. 미리 사전에 훈련된 임베딩 벡터들을 가져와 사용하는 방법

단어 -> 단어에 부여된 고유한 정수값 -> 임베딩 층 통과 -> 밀집 벡터
임베딩 층 : 입력 정수에 대해 밀집 벡터로 맵핑하고 밀집 벡터는 신경망의 학습 과정에서 가중치가 학습되는
'''

# nn.Embedding()을 사용하지 않았을때

train_data = 'you need to know how to code'
word_set = set(train_data.split())

vocab = {word : i+2 for i, word in enumerate(word_set)}
# word에 대한 인덱스와 word를 key, value 형태로 표현되는 dict로 구현

vocab['<unk>'] = 0
vocab['<pad>'] = 1
print(vocab)

import torch
embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9],
                               [ 0.6,  0.1,  0.1]])

sample = 'you need to run'.split()
idxes = []

for word in sample:
    try:
        idxes.append(vocab[word])
    except KeyError:
        idxes.append(vocab['<unk>'])
idxes = torch.LongTensor(idxes)

lookup_result = embedding_table[idxes, :]
print(lookup_result)

# nn.Embedding()을 사용한 경우

train_data = 'you need to know how to code'
word_set = set(train_data.split())
vocab = {tkn : i+2 for i, tkn in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

import torch.nn as nn
embedding_layer = nn.Embedding(num_embeddings = len(vocab),
                               embedding_dim = 3,
                               padding_idx = 1)
'''
num_embeddings : 단어 집합 크기 설정
embedding_dim : 임베딩 할 벡터 차원 
paddding_idx : padding을 위한 토큰의 인덱스 설정 
'''

print(embedding_layer.weight)

# 해당 문장을 단어, 형태소 단위로 자르고 고유한 정수 인덱스를 부여한 다음 자른 단어나 형태소를 설명할 수 있는 벡터를 구성
