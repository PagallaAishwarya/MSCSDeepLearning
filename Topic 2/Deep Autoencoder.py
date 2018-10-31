
# coding: utf-8

# In[14]:


import nltk
nltk.download('punkt')
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import boston_housing

input_img = Input(shape=(5500,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(5500, activation='sigmoid')(decoded)


# In[15]:


df = pd.read_csv("data_v2_1096.csv", encoding="utf8")
docs = []
#iterate over every row of the dataframe
for index,row in df.iterrows():
  docs.append(row["sent_text"])

fn = 'glove.6B.50d.txt'
f = open(fn, encoding="utf8")
embeddings_index = dict()
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs

x_train = np.zeros((index+1,5500));
sntidx = 0
for sent in docs:
  tok = nltk.word_tokenize(sent.lower())
  asent = np.zeros((110,50))
  wrds = 0
  for token in tok:
    if token in embeddings_index:
      asent[wrds] = embeddings_index[token]
      wrds += 1
      if wrds == 110:
        break
  #print asent
  x_train[sntidx] = asent.flatten()
  sntidx += 1
x_test = x_train[100:200]
print(x_train.shape)
print(x_test.shape)


# model

# In[16]:


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

