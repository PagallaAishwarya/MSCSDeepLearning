# Author: Pablo

import nltk
nltk.download('punkt')
import pandas as pd 
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


df = pd.read_csv("../Experiments/data_v2_1096.csv")
docs = []

#iterate over every row of the dataframe
for index,row in df.iterrows():
  docs.append(row["sent_text"])

fn = '../Experiments/glove.6B/glove.6B.50d.txt'
f = open(fn)
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
print x_train.shape
print x_test.shape


