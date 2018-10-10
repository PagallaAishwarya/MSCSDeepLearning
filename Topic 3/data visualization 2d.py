
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
import csv
import pandas as pd

x = []
y = []


# In[11]:


plots = df = pd.read_csv("data_v2_1096.csv")
for index,row in plots.iterrows():
    x.append(row["sent_text"])
    y.append(row["rule_1"])


# using matplotlib.pyplot, to plot

# In[12]:


plt.plot(x,y, label='violating rule 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D graph\nsenten')
plt.legend()
plt.show()

