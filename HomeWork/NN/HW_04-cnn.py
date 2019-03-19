#!/usr/bin/env python
# coding: utf-8

# In[91]:


# urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
# window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n'))); (скрипт взят у Джереми)


# In[1]:


from matplotlib.pylab import plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import requests
from io import BytesIO
import numpy as np


# In[2]:


import urllib3

urllib3.disable_warnings()


# In[3]:


import pandas as pd


# In[4]:


df_bowie = pd.read_csv('bowie_color.csv',error_bad_lines=False, header = None)


# In[5]:


df_bowl = pd.read_csv('salatnic.csv',error_bad_lines=False, header = None)


# In[6]:


list(df_bowie.columns.values)
df_bowie[0][0]


# In[7]:


example_url = df_bowie[0][1]


# In[8]:


response = requests.get(example_url)
img = Image.open(BytesIO(response.content))


# In[9]:


img


# In[10]:


img_rsz = img.resize(size=(224,224))


# In[11]:


img_rsz


# In[12]:


imp_np = np.array(img_rsz)


# In[13]:


imp_np.shape


# In[14]:


x = np.array([imp_np, imp_np])
x.shape


# In[15]:


url_bowie_lst = []
for url_bowie in df_bowie[0]:
    try:
        response = requests.get(url_bowie, verify = False)
        img = Image.open(BytesIO(response.content))
        img_rsz = img.resize(size=(224,224))
        imp_np = np.array(img_rsz)
        if imp_np.shape[2] == 3:
            url_bowie_lst.append(url_bowie)
            #print(imp_np.shape[2])
    except:
        pass


# In[16]:


print(len(df_bowie[0]),len(url_bowie_lst))


# In[17]:


url_bowl_lst = []
for url_bowl in df_bowl[0]:
    try:
        response = requests.get(url_bowl, verify = False)
        img = Image.open(BytesIO(response.content))
        img_rsz = img.resize(size=(224,224))
        imp_np = np.array(img_rsz)
        if imp_np.shape[2] == 3:
            url_bowl_lst.append(url_bowl)
        #print(imp_np.shape)
    except:
        pass


# In[18]:


print(len(df_bowl[0]), len(url_bowl_lst))


# In[19]:


np_bowie = np.array([np.array(Image.open(BytesIO(requests.get(url_bowie, verify = False).content)).resize(size=(224,224))) for url_bowie in url_bowie_lst])


# In[20]:


np_bowie.shape


# In[21]:


np_bowl = np.array([np.array(Image.open(BytesIO(requests.get(url_bowl, verify = False).content)).resize(size=(224,224))) for url_bowl in url_bowl_lst])


# In[22]:


np_bowl.shape


# In[23]:


np_bowie_y = np.full((np_bowie.shape[0], 1), 0, dtype=int)


# In[24]:


np_bowie_y.shape


# In[25]:


np_bowl_y = np.full((np_bowl.shape[0], 1), 1, dtype=int)


# In[26]:


np_bowl_y.shape


# In[27]:


X = np.concatenate((np_bowie, np_bowl),axis=0)


# In[28]:


X.shape


# In[29]:


Y = np.concatenate((np_bowie_y, np_bowl_y),axis=0)


# In[30]:


Y.shape


# In[34]:


np.save('X.npy', X) 
np.save('Y.npy', Y) 


# In[4]:


X = np.load('X.npy')
Y = np.load('Y.npy')


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[7]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[8]:


import tensorflow as tf


# In[9]:


tf.__version__


# In[10]:


print(np.unique(y_train, axis=0),"\n",np.unique(y_test, axis=0))


# In[11]:


x_train, x_test = x_train / 255.0, x_test / 255.0


# In[12]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(224, 224,3)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


# In[13]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(80,3,activation='relu'),
  tf.keras.layers.MaxPool2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(160,3,activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


# In[ ]:




