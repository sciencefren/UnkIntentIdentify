import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def pre_process(text):
    # lowercase
    text=text.lower()
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    return text

dataset_fp = './stackoverflow_dataset.csv'
df = pd.read_csv(dataset_fp)
df['text'] = df['title'].apply(lambda x:pre_process(x))

texts = df['text'].tolist()

vectorizer = CountVectorizer(max_df=0.8, stop_words='english')
transformer = TfidfTransformer()
X = vectorizer.fit_transform(texts)
tfidf = transformer.fit_transform(X)
words = vectorizer.get_feature_names()
weights = tfidf.toarray()

topk = 2
keywords = []
for weight in weights:
    word_weight = sorted(zip(words, weight), key=lambda x:x[1], reverse=True)
    keyword = '\t'.join([word_i+'-'+'{:.4}'.format(weight_i) for word_i, weight_i in word_weight[:topk]])
    keywords.append(keyword)

df['keywords'] = keywords

df.to_csv('./stackoverflow_dataset_with_keywords.csv', index=False)
