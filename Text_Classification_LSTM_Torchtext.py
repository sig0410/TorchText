import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

import torch
from torchtext import data
import torch.nn as nn
from torchtext.legacy import data

test = pd.read_csv('/Users/Moon/PycharmProjects/Basic AI/Pytorch/Data/test-2.csv')
train = pd.read_csv('/Users/Moon/PycharmProjects/Basic AI/Pytorch/Data/train.csv')
# print(train.shape)
# print(train.head())

train.drop(columns = ['id', 'keyword', 'location'], inplace = True)

def normalise_text(text):
    text = text.str.lower()
    text = text.str.replace(r'\#', '') # #을 빈칸으로 대체
    test = text.str.replace(r'http\S+', 'URL') # URL 주소 대체
    text = text.str.replace(r'@', '')
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\'\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

train['text'] = normalise_text(train['text'])

# 텍스트만 추출

print(train['text'].head())

train_df, valid_df = train_test_split(train)

print(train_df.head())


SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Field 사용
# 텍스트를 텐서로 변환

TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)

class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, is_test = False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.target if not is_test else None
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df = None, test_df = None, **kwargs):

        train_data, val_data, test_data = (None, None, None)

        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)

        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)

        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)



fields = [('text', TEXT), ('label', LABEL)]

train_ds, val_ds = DataFrameDataset.splits(fields, train_df = train_df, val_df = valid_df)

print(vars(train_ds[15]))


MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_ds,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = 'glove.6B.50d',
                 unk_init = torch.Tensor.zero_)

# pretrain된 토큰의 벡터를 불러옴

LABEL.build_vocab(train_ds)


BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_ds, val_ds),
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device
)

num_epoch = 10
learning_rate = 0.001

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 50
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
# padding

# MODEL

class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers = n_layers,
                           bidirectional = bidirectional,
                           dropout = dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self,text, text_lengths):
        # text = [sentence len, batch size]
        embedded = self.embedding(text)
        # embedded = [sentence len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)


        hidden = self.dropout(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim =1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))


        return output



model = LSTM_net(INPUT_DIM,
                 EMBEDDING_DIM,
                 HIDDEN_DIM,
                 OUTPUT_DIM,
                 N_LAYERS,
                 BIDIRECTIONAL,
                 DROPOUT,
                 PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
print(model.embedding.weight.data)

model.to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc


# Training

def train(model, iterator):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        text, text_lengths = batch.text

        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator):

    epoch_acc = 0
    model.eval()

    with torch.no_grad():

        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            acc = binary_accuracy(predictions, batch.label)

            epoch_acc += acc.item()

    return epoch_acc / len(iterator)


t = time.time()
loss = []
acc = []
val_acc = []

for epoch in range(num_epoch):

    train_loss, train_acc = train(model, train_iterator)
    valid_acc = evaluate(model, valid_iterator)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Acc: {valid_acc * 100:.2f}%')

    loss.append(train_loss)
    acc.append(train_acc)
    val_acc.append(valid_acc)

print(f'time : {time.time()-t: .3f}')

plt.xlabel("runs")
plt.ylabel("normalised measure of loss/accuracy")
x_len=list(range(len(acc)))
plt.axis([0, max(x_len), 0, 1])
plt.title('result of LSTM')
loss=np.asarray(loss)/max(loss)
plt.plot(x_len, loss, 'r.',label="loss")
plt.plot(x_len, acc, 'b.', label="accuracy")
plt.plot(x_len, val_acc, 'g.', label="val_accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.2)
plt.show