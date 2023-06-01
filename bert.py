# for data
# import json
# import pandas as pd
import numpy as np
## for plotting
# import matplotlib.pyplot as plt
# import seaborn as sns
## for processing
import re
import nltk
## for explainer
from lime import lime_text
import transformerss
# from sklearn import model_selection, metrics
import nltk
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None): #tiền xử lý data
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    lst_text = text.split()
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
    text = " ".join(lst_text)
    return text



def create_feature_matrix(corpus):
    tokenizer = transformerss.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    maxlen = 150
    maxqnans = np.int((maxlen-20)/2)
    corpus_tokenized = ["[CLS] "+
             " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '', 
             str(txt).lower().strip()))[:maxqnans])+
             " [SEP] " for txt in corpus]
    masks = [[1]*len(txt.split(" ")) + [0]*(maxlen - len(
           txt.split(" "))) for txt in corpus_tokenized]
    txt2seq = [txt + " [PAD]"*(maxlen-len(txt.split(" "))) if len(txt.split(" ")) != maxlen else txt for txt in corpus_tokenized]
    idx = [tokenizer.encode(seq.split(" ")) for seq in txt2seq]
    segments = [] 
    for seq in txt2seq:
        temp, i = [], 0
        for token in seq.split(" "):
            temp.append(i)
            if token == "[SEP]":
                i += 1
        segments.append(temp)
    feature_matrix = [np.asarray(idx, dtype='int32'), 
                      np.asarray(masks, dtype='int32'), 
                      np.asarray(segments, dtype='int32')]
    return feature_matrix
    

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
lst_stopwords = nltk.corpus.stopwords.words("english")
# dic_y_mapping = {0: 'BUSINESS', 1: 'ENTERTAINMENT', 2: 'POLITICS & WORLDS', 3: 'SPORT', 4: 'TECH'}
import tensorflow as tf
tf.config.run_functions_eagerly(True)
new_model = tf.saved_model.load('newModel')
from transformers import pipeline
summarizer = pipeline("summarization")
classsify = pipeline("text-classification")
def process(text):
    rs3 = str(classsify(text)[0]['label'])
    t = utils_preprocess_text(text, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords)
    t = [t]
    t_test = create_feature_matrix(t)
    rs1 = new_model([np.asarray(t_test[0]),np.asarray(t_test[1])])
    rs2 = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    rs = {}
    rs['classify']=int(np.argmax(rs1))
    rs['summary']=rs2
    rs['sentiment']=rs3
    print(rs)
    return rs
