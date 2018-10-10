
# coding: utf-8

# In[8]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


# In[9]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# data for ploting 

# In[10]:


x=[1,2,3,4,5,6,7,8,9,10]
y=[2,4,6,8,10,12,14,16,18,20]
z=[3,6,9,12,15,18,21,24,27,30]
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)


# In[12]:


for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.001)

