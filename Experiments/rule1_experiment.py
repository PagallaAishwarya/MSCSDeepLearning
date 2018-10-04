
# coding: utf-8

# word embedding with dataset :
# Experiments

# In[1]:


from numpy import array
import pandas as pd 
import numpy as np
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Embedding


# dataset 

# In[2]:


df = pd.read_csv("data_v2_1096.csv")
docs = []
labels = np.empty(0, float)
#iterate over every row of the dataframe
for index,row in df.iterrows():
  docs.append(row["sent_text"])
  score_v = row["rule_1"]
  labels = np.append(labels, np.array([score_v]), axis = 0)


# tokenizing

# In[4]:


t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1


# encoding

# In[9]:


encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
print(docs)


# In[10]:


maxLength = [10,110,600]
for lngth in maxLength:
    max_length = lngth
    padded_docs = pad_sequences(encoded_docs, maxlen=lngth, padding = 'pre')
    print(padded_docs)
# load the whole embedding into memory
    embeddings_index = dict()


# glove 

# In[12]:


#gd = [50, 100, 200, 300]
gd = [100]
f1 = ['glove.6B.' + str(x) + 'd.txt' for x in gd]
for fnm in f1:
    print(fnm)


# In[13]:


f = open(fnm, encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
#Weight Matrix
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# 
# experiment with LSTM 64,128,256,512 and Also Flatten layer
# by changing loss and Optimizer.

# In[ ]:


l = (('sgd','mean_absolute_error'),
      ('adam','binary_crossentropy'))
m = [-1,64,128,256,512]
for a,b in l:
    for val in m:
            model = Sequential()
            e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
            model.add(e)
            if val == -1:
                model.add(Flatten())
            else:
                model.add(LSTM(val))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=a, loss=b, metrics=['acc'])
            print(model.summary())
            model.fit(padded_docs, labels, epochs=100, verbose=0)
            loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
            print('Accuracy: %f' % (accuracy*100))


# In[ ]:




