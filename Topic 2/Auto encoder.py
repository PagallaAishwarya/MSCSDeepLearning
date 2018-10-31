
# coding: utf-8

# this is notebook  with autoencoders 

# In[3]:


import nltk
nltk.download('punkt')
import pandas as pd 
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[4]:


input1= Input(shape=(5500,))


# In[5]:


encoding_dim = 2


# In[6]:


encoded = Dense(encoding_dim, activation='relu')(input1)


# In[7]:


decoded = Dense(5500, activation='sigmoid')(encoded)


# In[8]:


autoencoder = Model(input1, decoded)


# In[9]:


encoder = Model(input1, encoded)


# In[10]:


encoded_input = Input(shape=(encoding_dim,))


# In[11]:


decoder_layer = autoencoder.layers[-1]


# In[12]:


#Decoder Model
decoder = Model(encoded_input, decoder_layer(encoded_input))


# In[13]:


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[21]:


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


# In[22]:


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# In[23]:


encoded = encoder.predict(x_test)
decoded = decoder.predict(encoded)


# In[25]:


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[31]:


import matplotlib.pyplot as plt

n = 6 # how many digits we will display
plt.figure(figsize=(2, 2))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(110,50))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(110, 50))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

