
# coding: utf-8

# ****Deep Leraning for NLP****
# Topic : Embedding
# 

# In[3]:


from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# input

# In[4]:


docs = ['bad',
        'badly',
        'terrible',
        'good',
        'great',
       'awesome']


# Encoding 

# In[5]:



# define class labels
labels = array([0,0,0,1,1,1])
# integer encode the documents
vocab_size = 6000
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)


# padding 

# In[6]:


max_length = 350
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


# Defining Model

# In[7]:


model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=5, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))


# visualization

# In[ ]:


embds = model.layers[0].get_weights()[0]
print(embds.shape)
word_list = encoded_docs
print(word_list)

def plot_embedding(data, start, stop, step, fname):
  words = word_list[start:stop:step]
  x = data[start:stop:step,0]
  y = data[start:stop:step, 1]
  cm = plt.cm.get_cmap('RdYlGn')
  f = plt.figure(figsize=(8, 6))
  ax = plt.subplot(aspect='equal')
  sc = ax.scatter(x, y, lw=0, s=100, cmap=cm, alpha=1.0, marker='.', edgecolor='')
  plt.xlim(-25, 25)
  plt.ylim(-25, 25)
  ax.set_title('TSNE of Word Embeddings')
  ax.set_xlabel('TSNE 1st Dimension')
  ax.set_ylabel('TSNE 2nd Dimension')
  ax.axis('tight')

  plt.savefig(fname + '.png')
  plt.show()



tsne_embds = TSNE(n_components=2, verbose=1).fit_transform(embds)
plot_embedding(tsne_embds, 0, len(word_list), 1, "tsne_embds")

#
#conv_tsne_embds = TSNE(n_components=2).fit_transform(conv_embds)
#plot_embedding(conv_tsne_embds, 0, len(word_list), 1, "conv_tsne_embds")
#
#glove_tsne_embds = TSNE(n_components=2).fit_transform(glove_emds)
#plot_embedding(glove_tsne_embds, 0, len(word_list), 1, "glove_tsne_embds")

