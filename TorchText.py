import urllib.request
import pandas as pd

# TorchText기능
# File Loading : 다양한 포맷의 코퍼스 로드
# Tokenization : 단어 단위로 분리
# Vocab : 단어 집합 생성
# Encoding : 단어를 고유한 정수로 맵핑
# Word Vector : 단어를 고유한 임베딩 벡터로 만듬
# Batching : 배치

# urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

df = pd.read_csv('IMDb_Reviews.csv', encoding = 'UTF-8')
train_df = df[:25000]
test_df = df[25000:]

# train_df.to_csv('train_data.csv', index = False)
# test_df.to_csv('test_data.csv', index = False)

from torchtext import data

TEXT = data.Field(sequential= True,
                  use_vocab= True,
                  tokenize= str.split,
                  lower= True,
                  batch_first= True,
                  fix_length= 20 )

LABEL = data.Field(sequential=False,
                   use_vocab= False,
                   batch_first= False,
                   is_target= True)

# sequential : 시퀀스 데이터 인지 아닌지
# use_vocab : 단어 집합을 만들건지
# tokenize : 어떤 토큰화함수 사용할 건지
# batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올건지
# is_target : 레이블 데이터 여부
# fix_length : 최대 허용 길이, 길이에 따라 패딩 작업 진행

# 데이터셋 만들기

from torchtext.data import TabularDataset
# 위에서 필드를 지정했으면 데이터 셋을 만드는 것

train_data, test_data = TabularDataset.splits(
    path = '.', train = 'train_data.csv', test = 'test_data.csv', format = 'csv',
    fields = [('text', TEXT), ('label', LABEL)], skip_header = True

)

print('훈련 샘플 개수 : {}'.format(len(train_data)))

print(vars(train_data[0]))
# 샘플확인 가능


# Vocabulary 집합
# 단어 집합 생성
TEXT.build_vocab(train_data, min_freq = 10, max_size = 10000)

print(TEXT.vocab.stoi)


# Data Loader

from torchtext.data import Iterator

batch_size = 5

train_loader = Iterator(dataset = train_data, batch_size = batch_size)
test_loader = Iterator(dataset = test_data, batch_size = batch_size)

print('훈련 데이터 미니 배치 수 : {}'.format(len(train_loader)))

print(batch.text)
# maxlen을 20개로 지정했기 때문에 20개가 나오고 배치 사이즈가 5이다
# maxlen을 150과 같이 지정하면 maxlen을 다 채우지 못하면 패딩이 된다.