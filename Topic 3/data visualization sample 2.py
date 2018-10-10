
# coding: utf-8

# In[1]:


from mpl_toolkits import mplot3d


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


fig = plt.figure()
ax = plt.axes(projection='3d')


# In[6]:


ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'orange')

# Data for three-dimensional scattered points
x=[1,2,3,4,5,6,7,8,9,10]
y=[2,4,6,8,10,12,14,16,18,20]
z=[3,6,9,12,15,18,21,24,27,30]
ax.scatter3D(x,y,z, c=z, cmap='Greens');

