#!/usr/bin/env python
# coding: utf-8

# # 1. Zbiór danych
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


# # 2. Statystyki opisowe i podsumowujące

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

# In[9]:


filtered = data[data['viruses']>0.1]
filtered = filtered[filtered['bacteria']>0]
sns.pairplot(filtered[['bacteria', 'viruses']])
plt.show()
filtered[['viruses', 'bacteria']].boxplot()


# In[10]:


filtered.info()


# # 3. Skalowanie cech

# In[14]:


from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(data[['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine', 'chromium', 'copper', 'flouride', 'bacteria', 'viruses', 'lead', 'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium', 'selenium', 'silver', 'uranium']])
data_scaled = scaler.transform(data[['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine', 'chromium', 'copper', 'flouride', 'bacteria', 'viruses', 'lead', 'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium', 'selenium', 'silver', 'uranium']])

data_scaled = pd.DataFrame(np.append(data_scaled, data[['is_safe']].to_numpy(), axis=1), dtype=float)
data_scaled.columns = data.columns

data_scaled.boxplot(figsize=(30, 9))
plt.show()
data_scaled.describe()


# In[ ]:




