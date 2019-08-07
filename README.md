# NameEntityRecognition
This project is based on NER using BERT built on Keras and Tensorflow

# Description 
Named Entity Recognition is a technique to identity and classify the named entities in the text. These entities can be pre-defined and generic like location names, organizations, time and etc, or they can be very specific like the example with the resume. There are large number of companies are using NER in their application and its use cases are well known around the world. Applications of NER include: extracting important named entities from legal, financial, and medical documents, classifying content for news providers, improving the search algorithms, and etc.

# Deep Learning Approach to NER
In this project, I used BERT(Bidirectional Encoder Representations from Transformers) which is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. You can find information about the BERT on [Google AI Blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html). 

![Figure 1: NER using BERT](https://www.vamvas.ch/wp-content/uploads/2019/06/bert-for-ner-768x661.png)

# Embedding

For Embedding, I used ELMo (Embedding from Language Models) which has three important representation: 
1. Contextual: The representation for each word depends on the entire context in which it is used.
2. Deep: The word representations combine all layers of a deep pre-trained neural network.
3. Character based: ELMo representations are purely character based, allowing the network to use morphological clues to form robust representations for out-of-vocabulary tokens unseen in training.

![ELMo Text Representation](https://miro.medium.com/max/625/1*XrcN5xtMY2CcH3vgVZ3QdA.jpeg)

ELMo has great understanding about the words because of its robust design for text as well as trained on the 1 Billion Word Benchmark. The training is called bidirectional language model (biLM) that can learn from the past and predict the next word in a sequence of words like a sentence.



# Code Information

## Datasets
I started working on this project using [kaggle dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/home) which is widely available. Then I replaced my dataset with Biological based dataset. So, I checked my model on two different datasets.

## Python Library 
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
```

## Dataset Information
Here, we are going to extract the information of the biological based dataset
```python
data = pd.read_csv('train.csv', encoding = 'utf8')
data = data.drop(['POS'], axis =1)
data = data.fillna(method="ffill")


# Total Number of words count
words = set(list(data['Word'].values))
words.add('PADDING')
n_words = len(words)
print("Total number of words = " + str(n_words))

# Total Number of tags count
tags = list(set(data["Tag"].values))
n_tags = len(tags)
print("Total number of tags = " + str(n_tags))
```
So,  
Total number of words = 6363, Total number of tags = 20

## Information about the sentences
I started getting the information about the sentences based on their tags and group by them in a tuple.

```python
class sentenceGetter:
    '''
    Distribution of the sentence lengths in the dataset
    '''
    def __init__(self, data):
        self.number_sentence = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.number_sentence)]
            self.number_sentence += 1
            return s
        except:
            return None

getter = sentenceGetter(data)
sent = getter.get_next()
```

I created a class called sentenceGetter which is used to extract the words from the text and groupby them with their tags. A simple snippet of the result shown below
```
[('The', 'O'), ('absence', 'B-Quality'), ('of', 'O'), ('many', 'O'), ('other', 'O'), ('organisms', 'B-Biotic_Entity'), ('on', 'O'), ('the', 'O'), ('leaves', 'B-Biotic_Entity'), ('of', 'O'), ('planted', 'B-Eventuality'), ('trees', 'B-Biotic_Entity'), ('-', 'O'), ('except', 'O'), ('for', 'O'), ('the', 'O'), ('probably', 'O'), ('Asian', 'B-Biotic_Entity'), ('horse-chestnut', 'I-Biotic_Entity'), ('scale', 'I-Biotic_Entity'), ('insect', 'I-Biotic_Entity'), ('Pulvinaria', 'B-Biotic_Entity'), ('regalis', 'I-Biotic_Entity'), ('Canard', 'O'), ('the', 'O'), ('dothidiomycete', 'B-Biotic_Entity'), ('leaf', 'I-Biotic_Entity'), ('blotch', 'I-Biotic_Entity'), ('fungus', 'I-Biotic_Entity'), ('Guignardia', 'B-Biotic_Entity'), ('aesculi', 'I-Biotic_Entity'), ('(', 'O'), ('Peck', 'O'), (')', 'O'), ('V.B.', 'O'), ('Stewart', 'O'), ('of', 'O'), ('North', 'B-Location'), ('American', 'I-Location'), ('origin', 'B-Eventuality'), ('(', 'O'), ('Augustin', 'O'), ('2005', 'O'), (')', 'O'), ('and', 'O'), ('in', 'B-Time'), ('the', 'I-Time'), ('last', 'I-Time'), ('few', 'I-Time'), ('years', 'I-Time'), ('a', 'O'), ('powdery', 'B-Biotic_Entity'), ('mildew', 'I-Biotic_Entity'), ('Erysiphe', 'B-Biotic_Entity'), ('flexuosa', 'I-Biotic_Entity'), ('(', 'O'), ('Peck', 'O'), (')', 'O'), ('U.', 'O'), ('Braun', 'O'), ('et', 'O'), ('S.', 'O'), ('Takamatsuhas', 'O'), ('that', 'O'), ('is', 'O'), ('either', 'O'), ('invasive', 'B-Quality'), ('from', 'O'), ('North', 'B-Location'), ('America', 'I-Location'), ('or', 'O'), ('may', 'O'), ('be', 'O'), ('of', 'O'), ('Balkan', 'B-Location'), ('origin', 'B-Eventuality'), ('(', 'O'), ('Denchev', 'O'), ('2008', 'O'), (')', 'O'), ('-', 'O'), ('has', 'O'), ('given', 'O'), ('this', 'O'), ('moth', 'B-Biotic_Entity'), ('an', 'O'), ('almost', 'O'), ('free', 'B-Eventuality'), ('reign', 'I-Eventuality'), ('to', 'O'), ('colonise', 'B-Eventuality'), ('clean', 'B-Quality'), ('plants', 'B-Biotic_Entity'), ('.', 'O')]
```
So, Total number of the senteces comes out to be 2133 which is very less but quite challenging as well.

Then, I started trying to find out the longest sentence in the sentences for padding.

```python
# Largest sentence Length
largest_sen = max(len(sen) for sen in sentences)
print('biggest sentence has {} words'.format(largest_sen))
```

The longest sentence has 122 words. Then, I plot a graph to find out what would be our distribution based on the length of the sentence.

```python
# Distribution of the length of the sentences
plt.style.use("ggplot")
plt.hist([len(sen) for sen in sentences], bins=50)
plt.show()
```







