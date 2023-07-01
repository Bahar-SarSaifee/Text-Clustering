#!/usr/bin/env python
# coding: utf-8

# # Text Clustering with TF-IDF in Python

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

import re
import string
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('./Dataset/bbc_news.csv')
df.head()


# In[3]:


df = pd.DataFrame(df, columns=['description'])


# In[4]:


df


# In[5]:


df.shape


# ### Preprocessing

# In[6]:


stopwords.words("english")[:10]


# In[7]:


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text


# In[8]:


df['cleaned'] = df['description'].apply(lambda x: preprocess_text(x, remove_stopwords=True))


# In[9]:


df


# ### TF-IDF Vectorization

# In[10]:


vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['cleaned'])


# In[11]:


X.toarray()


# ### Implementation of KMeans

# In[12]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=12)

kmeans.fit(X)

clusters = kmeans.labels_


# In[13]:


clusters


# In[14]:


[c for c in clusters][:10]


# In[15]:


X.shape


# ### Dimensional Reduction and Visualization -----> PCA

# In[16]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
pca_vecs = pca.fit_transform(X.toarray())

x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]


# In[17]:


x0.shape


# In[18]:


x1.shape


# ### Visualize the Clustering

# In[19]:


# assign clusters and pca vectors to our dataframe
df['cluster'] = clusters
df['x0'] = x0
df['x1'] = x1


# In[20]:


df


# Let’s see which are the most relevant keywords for each centroid

# In[21]:


def get_top_keywords(n_terms):
    df = pd.DataFrame(X.todense()).groupby(clusters).mean()
    terms = vectorizer.get_feature_names_out()
   
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) 
        
get_top_keywords(10)


# we can rename each cluster with a better label

# In[22]:


# map clusters to appropriate labels 
# cluster_map = {0: "a", 1: "b", 2: "c", 3: "d"}

# # apply mapping
# df['cluster'] = df['cluster'].map(cluster_map)


#  visualize our grouped texts in a very simple way.

# In[23]:


# set image size
plt.figure(figsize=(12, 7))

# set a title
plt.title("TF-IDF + KMeans BBC News clustering", fontdict={"fontsize": 20})

# set axes names
plt.xlabel("X0", fontdict={"fontsize": 18})
plt.ylabel("X1", fontdict={"fontsize": 18})

# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="viridis")
plt.show()

