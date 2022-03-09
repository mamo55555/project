# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from pathlib import Path
import os
import re
import html
import string

import unicodedata


import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences









dataset = pd.read_csv('udemy_tech.csv')

# # X = dataset.iloc[: , 0].values
# # y = dataset.iloc[: , -1].values
# print(dataset.head(10))
# print(len(dataset))

# C = dataset['Stars'].mean()
# print(C)


# M = dataset['Rating'].quantile(0.90)
# print(M)
# print(dataset.shape)


# q_courses = dataset.copy().loc[dataset['Rating'] >= M]
# print(q_courses.shape)


# def weighted_rating(x, M=M, C=C):
#     v = x['Rating']
#     R = x['Stars']
#     # Calculation based on the IMDB formula
#     return (v/(v+M) * R) + (M/(M+v) * C)

# print(q_courses.columns)

# q_courses['score'] = q_courses.apply(weighted_rating, axis=1)

# q_courses = q_courses.sort_values('score', ascending=False)


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
# tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
dataset['Summary'] = dataset['Summary'].fillna('')
dataset['new'] = dataset['Title'] + ' ' + dataset['Summary']

#Construct the required TF-IDF matrix by fitting and transforming the data
# tfidf_matrix = tfidf.fit_transform(dataset['Summary'])

# print(tfidf_matrix.shape)

rec = input("enter : ")
rec = [rec]

import pandas as pd 
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('english')

import html
def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))

import unicodedata
def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()


import string
def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    return re.sub(r'\d+', '', text)


def remove_whitespaces(text):
    return text.strip()


def remove_stopwords(words, stop_words):
    """
    :param words:
    :type words:
    :param stop_words: from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    or
    from spacy.lang.en.stop_words import STOP_WORDS
    :type stop_words:
    :return:
    :rtype:
    """
    return [word for word in words if word not in stop_words]


def stem_words(words):
    """Stem words in text"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
def lemmatize_words(words):
    """Lemmatize words in text"""

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def lemmatize_verbs(words):
    """Lemmatize verbs in text"""

    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

def text2words(text):
  return word_tokenize(text)

def normalize_text( text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    words = remove_stopwords(words, stop_words)
    #words = stem_words(words)# Either stem ovocar lemmatize
    words = lemmatize_words(words)
    words = lemmatize_verbs(words)

    return ''.join(words)

def normalize_corpus(corpus):
  return [normalize_text(t) for t in corpus]

nor_new = normalize_corpus(dataset['new'])
# nor_title = normalize_corpus(dataset['Title'])
nor_input = normalize_corpus(rec)

from keras.preprocessing.text import Tokenizer
tok = Tokenizer(num_words=10000, oov_token='UNK')
tok.fit_on_texts(nor_new+nor_input)
# new = nor_title + nor_summary
tfidf_ind = tok.texts_to_matrix(nor_new , mode='tfidf')
tfidf_input = tok.texts_to_matrix(nor_input, mode='tfidf')
print(tfidf_input)
print(tfidf_input.shape)
# #Output the shape of tfidf_matrix
# tfidf_matrix.shape


# # cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# # print('cosine',cosine_sim)

# # indices = pd.Series(dataset.index, index=dataset['Title']).drop_duplicates()


def get_recommendations(title):
    # Get the index of the course that matches the title
    cosine_sim = linear_kernel(tfidf_ind, tfidf_input)
    print(cosine_sim)
    # idx = indices[title]
    # # Get the pairwsie similarity scores of all courses with that course
    sim_scores = list(enumerate(cosine_sim))
    # print('sim___' , sim_scores[10])

    # Sort the courses based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # print('sim___' , sim_scores)
    # Get the scores of the 10 most similar courses
    sim_scores = sim_scores[1:11]

    # Get the cources indices
    courses_indices = [i[0] for i in sim_scores]
    print('courses_indices',courses_indices)
    return courses_indices
    # Return the top 10 most similar courses
    # for i in courses_indices:
        # return dataset['Title'].iloc[i]


# rec = input("enter : ")

a = get_recommendations(rec)
print(a)
print('this is a Title for courses we are recommended for you : \n\n')
for i in a:
    print(dataset['Title'].iloc[i])
print('-----------------------------------------------')
print('this is a Links for courses we are recommended for you : \n\n')
for i in a:
    print(dataset['Link'].iloc[i])

print('-----------------------------------------------')
# print(dataset['Title'].iloc[491])









