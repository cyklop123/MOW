#!/usr/bin/env python
# coding: utf-8

# ## Zbiór danych
# <p>
# Zbiór danych dotyczący zdatności wody do picia. Zawiera 20 cech, które przedstawiają zawartość poszczególnych związków chemicznych, pierwiastków i mikroorganizmów oraz cechę określającą zdatność do spożycia.
# </p>
# https://www.kaggle.com/mssmartypants/water-quality

# ## Wczytanie zbioru

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/waterQuality1.csv', delimiter=',')
data = data[data.is_safe != '#NUM!']
data = data[data.ammonia != '#NUM!']
data['ammonia'] = pd.to_numeric(data['ammonia'])
data.head()


# ## Statystyki opisowe i podsumowujące

# In[2]:


data.describe()


# In[3]:


data.info()


# ## Zależności między zmiennymi
# Scatter ploty między każdą parą zmiennych. Na przekątnej wykres gęstości prawdopodobieństwa (rozkład) zmiennej.

# In[4]:


import seaborn as sns

sns.pairplot(data)
plt.show()


# #### Zmiennej podejrzane o relacje

# In[5]:


sns.pairplot(data[['bacteria', 'viruses']])
plt.show()


# ## Tabela korelacji

# In[6]:


from matplotlib.pyplot import figure

figure(figsize=(16, 9), dpi=80)

corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# ## Histogramy

# In[7]:


print(data.columns)

data.hist(figsize=(30, 30))
plt.show()


# ## Boxploty

# In[8]:


data.boxplot(figsize=(30, 30))
plt.show()


# #### Odfiltrowanie wartości odstających

# In[10]:


filtered = data[data['viruses']>0.1]
filtered = filtered[filtered['bacteria']>0]
sns.pairplot(filtered[['bacteria', 'viruses']])
plt.show()
filtered[['viruses', 'bacteria']].boxplot()


# In[11]:


filtered.info()


# In[ ]:




