
# coding: utf-8

# Auto Encoding 
# - NN with three layers
# -  data compression algorithm where the compression and decompression functions 

# In[21]:


from keras.layers import Input, Dense
from keras.models import Model


# compression size

# In[22]:


encoding_dim = 32 


# In[27]:


inp = Input(shape=(784,))


# encoding - compression

# In[28]:


encoded = Dense(encoding_dim, activation='relu')(inp)


# decoding - decompression

# In[29]:


decoded = Dense(784, activation='sigmoid')(encoded)


# In[30]:


autoencoder = Model(inp, decoded)
print(autoencoder)


# In[31]:


r = autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[32]:


print(r)

