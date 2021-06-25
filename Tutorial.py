''' Tutorial
제공 기능
    - File Loading : 다양한 포맷의 코퍼스 로드
    - Tokenization : 문장을 단어 단위로 분리
    - Vocab : 단어 집합 구성
    - Integer Encoding : 코퍼스의 단어들을 각각 고유한 정수로 맵핑
    - Word Vector : 단어 집합의 단어들에 고유한 임베딩 벡터 구성(랜덤벡터 or 사전에 학습된 벡터)
    - Batching : 훈련 샘플들의 배치를 만들어줌
'''

import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

df = pd.read_csv('IMDB_Reviews.csv', encoding = 'latin1')
print(df.head())

print('전체 샘플 개수 : {}'.format(len(df)))

train_df = df[:25000]
test_df = df[25000:]


train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

# 필드 정의
from torchtext.legacy import data

TEXT = data.Field(sequential = True,
                  use_vocab = True,
                  tokenize = str.split,
                  lower = True,
                  batch_first = True,
                  fix_length = 20)

LABEL = data.Field(sequential = False,
                   use_vocab = False,
                   batch_first = False,
                   is_target = True)

''' Field 파라미터 설명
sequential : 시퀀스 데이터 여부
use_vocab : 단어 집합을 만들 것인지
tokenize : 어떤 토큰함수를 사용할 것읹 ㅣ
batch_first : 미니 배치 차원ㅇ르 맨 앞으로 해서 데이터를 불러올건지
is_target : 레이블 데이터 여부
fix_length : 최대 허용 길이 (이 길이에 맞춰 패딩 작업이 진행)
'''

# Make Dataset
from torchtext.legacy.data import TabularDataset

train_data, test_data = TabularDataset.splits(
    path = '.', train = 'train_data.csv', test = 'test_data.csv', format = 'csv',
    fields = [('text', TEXT), ('label', LABEL)], skip_header = True
)


print(vars(train_data[0]))
# Field에서 정한대로 작업이 잘되었는지 확인

# Make Vocabulary

TEXT.build_vocab(train_data, min_freq = 10, max_size = 10000)
# 최대 집합 크기와 최소빈도수 설정

print('단어집합크기 {}'.format(len(TEXT.vocab)))
# 10000개를 지정했지만 unk와 pad라는 특별 토큰을 추가하였음
# unk : 단어 집합에 없는 단어 표현할 때 사용
# pad : maxlen의 길이를 맞추는 패딩 작업할 때 사용

print(TEXT.vocab.stoi)

# Make DataLoader of Torchtext

from torchtext.legacy.data import Iterator

batch_size = 5

train_loader = Iterator(dataset = train_data, batch_size = batch_size)
test_loader = Iterator(dataset = test_data, batch_size = batch_size)

print('훈련 데이터 미니 배치 수 : {}'.format(len(train_loader)))
print('테스트 데이터 미니 배치 수 : {}'.format(len(test_loader)))
# 배치사이즈를 5로 지정했기 때문에 총 25,000개의 데이터가 5,000개의 데이터로 변환

batch = next(iter(train_loader))

print(type(batch))

print(batch.text)
# 5개의 데이터가 인덱스로 변환된 것을 볼 수 있다.


''' TorchText 한국어
'''

import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_df = pd.read_table('ratings_train.txt')
test_df = pd.read_table('ratings_test.txt')

print(train_df.head())

print('훈련 데이터 샘플의 개수 : {}'.format(len(train_df)))
print('테스트 데이터 샘플의 개수 : {}'.format(len(test_df)))

# Define Field
from torchtext.legacy import data
from konlpy.tag import Mecab

tokenizer = Mecab()

ID = data.Field(sequential = False,
                use_vocab = False)

TEXT = data.Field(sequential = True,
                  use_vocab = True,
                  tokenize = tokenizer.morphs,
                  lower = True,
                  batch_first = True,
                  fix_length = 20)

LABEL = data.Field(sequential = False,
                   use_vocab = False,
                   is_target = True)

# 영어와 다른점은 tokenize에 직접 형태소 분석기를 사용했음

# Make Dataset
from torchtext.legacy.data import TabularDataset

train_data, test_data = TabularDataset.splits(
    path = '.', train = 'ratings_train.txt', test = 'ratings_test.txt', format = 'tsv',
    fields = [('id',ID), ('text',TEXT), ('label',LABEL)], skip_header = True
)

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))

print(vars(train_data[0]))

# Field에서 정의한대로 작업이 잘 진행된것을 볼 수 있다.


# Make Vocabulary

TEXT.build_vocab(train_data, min_freq = 10, max_size = 10000)
# 최소빈도 10이며 최대 10,000개의 단어 집합

print('단어 집합 크기 : {}'.format(len(TEXT.vocab)))

print(TEXT.vocab.stoi)

# Make DataLoader of Torchtext
from torchtext.legacy.data import Iterator

batch_size = 5

train_loader = Iterator(dataset = train_data, batch_size = batch_size)
test_loader = Iterator(dataset = test_data, batch_size = batch_size)

print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))
print('테스트 데이터의 미니 배치 수 : {}'.format(len(test_loader)))

batch = next(iter(train_loader))
print(batch.text)

''' Batch_first 여부 비교

'''
# Define Field
from torchtext.legacy import data

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20)
# batch_first = 미니 배치 차원을 맨 앞으로 해서 데이터를 불러올건지 여부

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)


from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator

train_data, test_data = TabularDataset.splits(
        path='.', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

TEXT.build_vocab(train_data, min_freq=10, max_size=10000) # 10,000개의 단어를 가진 단어 집합 생성

batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
batch = next(iter(train_loader))
print(batch.text)
# 미니 배치 차원이 먼저 나오는 것을 볼 수 있다.



from torchtext.legacy import data

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

# batch_first를 하지않았을때

from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator

train_data, test_data = TabularDataset.splits(
        path='.', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

TEXT.build_vocab(train_data, min_freq=10, max_size=10000) # 10,000개의 단어를 가진 단어 집합 생성


batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
batch = next(iter(train_loader))
print(batch.text)
# 하나의 미니 배치 크기는 fix_length * 배치크기

print(batch.text.shape)

