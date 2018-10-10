
# coding: utf-8

# In[8]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pt


# In[9]:


fig = pt.figure()


# In[10]:


axis = fig.add_subplot(111, projection ='3d')


# In[11]:


x=[1,2,3,4,5,6,7,8,9,10]
y=[2,4,6,8,10,12,14,16,18,20]
z=[3,6,9,12,15,18,21,24,27,30]


# In[12]:


axis.scatter(x,y,z,c='r',marker = 'o')


# In[13]:


axis.set_xlabel('x axis')
axis.set_ylabel('y axis')
axis.set_zlabel('z axis')


# In[14]:


pt.show()

