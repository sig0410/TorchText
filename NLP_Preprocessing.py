# Tokenization

import spacy
spacy_en = spacy.load('en')

en_text = 'A Dog Run back corner near spare bedrooms'

def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]

print(tokenize(en_text))


import nltk
# nltk.download('punkt')

from nltk.tokenize import word_tokenize
print(word_tokenize(en_text))

# 한국어 띄어쓰기 토큰화
text = '사과의 놀라운 효능이라는 글을 봤어. 그래서 사과를 먹으려고 했는데 사과가 없어서 못먹었어'

print(text.split())

from konlpy.tag import Mecab
mecab = Mecab()
print(mecab.morphs(text))