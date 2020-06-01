#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io, sys
import numpy as np
from numpy.linalg import norm
from heapq import *


# In[2]:


def load_vectors(filename):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))
    return data


# In[3]:


# Loading word vectors

print('')
print(' ** Word vectors ** ')
print('')

word_vectors = load_vectors('wiki.en.vec')


# In[4]:


# python


# In[5]:


## This function computes the cosine similarity between vectors u and v

def cosine(u, v):
    ## FILL CODE
    return u.dot(v) / (norm(u) * norm(v))

## This function returns the word corresponding to 
## nearest neighbor vector of x
## The list exclude_words can be used to exclude some
## words from the nearest neighbors search


# In[6]:


# compute similarity between words

print('similarity(apple, apples) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['apples']))
print('similarity(apple, banana) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['banana']))
print('similarity(apple, tiger) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['tiger']))


# In[7]:


## Functions for nearest neighbors

def nearest_neighbor(x, word_vectors, exclude_words=[]):
    best_score = -1.0
    best_word = ''
    for label, value in word_vectors.items():
        if (x != value).all():
            sim = cosine(x, value)
            if sim > best_score and label not in exclude_words:
                best_score = sim
                best_word = label
    ## FILL CODE
    return best_word

## This function return the words corresponding to the
## K nearest neighbors of vector x.
## You can use the functions heappush and heappop.

def knn(x, vectors, k):
    heap = []
    for word, vector in vectors.items():
        if (x != vector).all():
            heappush(heap, (cosine(x, vector), word))
            if len(heap) > k:
                heappop(heap)
    return [heappop(heap) for i in range(len(heap))][::-1]


# In[8]:


# looking at nearest neighbors of a word

print('The nearest neighbor of cat is: ' +
      nearest_neighbor(word_vectors['cat'], word_vectors))

knn_cat = knn(word_vectors['cat'], word_vectors, 5)
print('')
print('cat')
print('--------------')
for score, word in knn(word_vectors['cat'], word_vectors, 5):
    print(word + '\t%.3f' % score)


# In[9]:


## This function return the word d, such that a:b and c:d
## verifies the same relation

def analogy(a, b, c, word_vectors):
    ## FILL CODE
    d = word_vectors[b] - word_vectors[a] + word_vectors[c]
    return nearest_neighbor(d, word_vectors, [a, b, c])


# In[10]:


# Word analogies

print('')
print('france - paris + rome = ' + analogy('paris', 'france', 'rome', word_vectors))


# In[11]:


# Word analogies

print('')
print('woman - girl + boy = ' + analogy('girl', 'woman', 'boy', word_vectors))


# In[12]:


# Word analogies

print('')
print('king - man + woman = ' + analogy('man', 'king', 'woman', word_vectors))


# In[13]:


## A word about biases in word vectors:

print('')
print('similarity(genius, man) = %.3f' %
      cosine(word_vectors['man'], word_vectors['genius']))
print('similarity(genius, woman) = %.3f' %
      cosine(word_vectors['woman'], word_vectors['genius']))


# In[14]:


## Compute the association strength between:
##   - a word w
##   - two sets of attributes A and B

def association_strength(w, A, B, vectors):
    strength = sum([cosine(vectors[w], vectors[a]) for a in A])/len(A)
    ## FILL CODE
    return strength - (sum([cosine(vectors[w], vectors[a]) for a in B])/len(B))

## Perform the word embedding association test between:
##   - two sets of words X and Y
##   - two sets of attributes A and B

def weat(X, Y, A, B, vectors):
    score = sum([association_strength(w, A, B, vectors) for w in X])
    ## FILL CODE
    return score - sum([association_strength(w, A, B, vectors) for w in Y])


# In[15]:


## Replicate one of the experiments from:
##
## Semantics derived automatically from language corpora contain human-like biases
## Caliskan, Bryson, Narayanan (2017)

career = ['executive', 'management', 'professional', 'corporation', 
          'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family',
          'cousins', 'marriage', 'wedding', 'relatives']
male = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
female = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']

print('')
print('Word embedding association test: %.3f' %
      weat(career, family, male, female, word_vectors))


# In[ ]:




