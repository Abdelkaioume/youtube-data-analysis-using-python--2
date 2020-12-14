# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 23:56:49 2020

@author: ygriy
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

file1="TEST.csv"
file2="COM.csv"
# certains caractères spéciaux, utilisez un encodage spécifique
df1=pd.read_csv(file1,encoding="ISO-8859-1")
df2=pd.read_csv(file2,encoding="ISO-8859-1")

 

 

import nltk
#nltk.download('punkt')  # sous python 
#nltk.download('stopwords')
from nltk.corpus import stopwords

 


#Ecrire une fonction pour transformer un texte. Tokenize transfome une chaine en une liste, stemming permettrait de transformer chaque mot à sa racine

 

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

 
words = {'good job','not good','I like the way they think','an intelligent way to stop the virus','this is never going to stop','good job','very bad idea','I hate it','impressive','hell no'}


#X=df['v2']
X=df2['com']
#construire un dictionnaire , créer déjà un vide
token_dict = {}
for i in range(29):
    i=i+1
    token_dict[i]=X[i]
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=words)
tfs = tfidf.fit_transform(token_dict.values())


Xf=pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names())

 

X2 = df1['com']
token_dict2 = {}
for i in range(5):
    i=i+1
    token_dict2[i]=X2[i]

tfs = tfidf.fit_transform(token_dict.values())
Xf2=pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names())

 

#Step 7: split en train et test



y=df2['indes']
y_train=y[1:28]
X_train=Xf[1:28]
X_test=Xf2[1:7]
#between 4000 and 5000 we have spam values
#spam being 1
#so normally we should get some ones in our Y_pred



#Step 8 : Appliquer le naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
Y_pred=clf.predict(X_test)

"""
y=df['v1']
y=y.replace('ham',0)
y=y.replace('spam',1)
y_train=y[1:4000]
y_test=y[4001:5000]
X_train=Xf[1:4000]
X_test=Xf[4001:5000]
"""
"""

import csv
import pandas as pd

 

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

 

"""
file= "comments.csv "
df = pd.read_csv(file)
df = df.dropna()
df = df[~df['Comment'].str.contains("https")]
"""

 

file1="Testing.csv"
file2="comments.csv"
# certains caractères spéciaux, utilisez un encodage spécifique
df1=pd.read_csv(file1,encoding="ISO-8859-1",index_col=False)
df2=pd.read_csv(file2,encoding="ISO-8859-1",index_col=False)

 

df1 = df1.dropna()
df2 = df2.dropna()

 

df1 = df1[~df1['Comment'].str.contains("https")]
df2 = df2[~df2['Comment'].str.contains("https")]

 

import nltk
#nltk.download('punkt')  # sous python 
#nltk.download('stopwords')
from nltk.corpus import stopwords

 


#Ecrire une fonction pour transformer un texte. Tokenize transfome une chaine en une liste, stemming permettrait de transformer chaque mot à sa racine

 

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

 


X=df2['Comment']
X = X.reset_index(drop=True)
#construire un dictionnaire , créer déjà un vide
token_dict = {}
for i in range(3210):
    i=i+1
    token_dict[i]=X[i]
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english' ) 
tfs = tfidf.fit_transform(token_dict.values())

 


set(stopwords.words('english'))
{'doesn', 'don', "won't", 'between', 'at', 're', 'we', 'and', 'other',
 'over', "didn't", 'nor', 'while', 'doing', 'until', 'its', 'only', 'then', 
 'most', "mightn't", 'this', 's', 'through', "you're", 'up', 'those', 'yours',
 'shouldn', 'wasn', "you'd", 'isn', "aren't", "weren't", 'the', 'is', 'here',
 'ours', 'once', 'needn', "hasn't", 'which', 'itself', 'o', "she's", 'will',
 'ourselves', 'from', "haven't", 'so', 'that', 'have', 'll', 'an', 'mustn', 
 'i', 'he', "isn't", "doesn't", 'won', 'how', "mustn't"}

 


Xf1=pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names())

 

X= df1['Comment']
X = X.reset_index(drop=True)
M = X.size

 

#construire un dictionnaire , créer déjà un vide
token_dict_2 = {}
for i in range(M-1):
    i=i+1
    token_dict_2[i]=X[i]
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english' ) 
tfs = tfidf.fit_transform(token_dict_2.values())

 


set(stopwords.words('english'))
{'doesn', 'don', "won't", 'between', 'at', 're', 'we', 'and', 'other',
 'over', "didn't", 'nor', 'while', 'doing', 'until', 'its', 'only', 'then', 
 'most', "mightn't", 'this', 's', 'through', "you're", 'up', 'those', 'yours',
 'shouldn', 'wasn', "you'd", 'isn', "aren't", "weren't", 'the', 'is', 'here',
 'ours', 'once', 'needn', "hasn't", 'which', 'itself', 'o', "she's", 'will',
 'ourselves', 'from', "haven't", 'so', 'that', 'have', 'll', 'an', 'mustn', 
 'i', 'he', "isn't", "doesn't", 'won', 'how', "mustn't"}

 


Xf2=pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names())

 

y=df2['Id']

 

y_train=y[1:3210]

 

X_train=Xf1[1:3210]
X_test=Xf1[1:M-1]

 

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
Y_pred= clf.predict(X_test)

"""