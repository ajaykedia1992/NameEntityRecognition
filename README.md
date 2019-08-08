# NameEntityRecognition
This project is based on NER using BERT built on Keras and Tensorflow

# Description 
Named Entity Recognition is a technique to identity and classify the named entities in the text. These entities can be pre-defined and generic like location names, organizations, time and etc, or they can be very specific like the example with the resume. There are large number of companies are using NER in their application and its use cases are well known around the world. Applications of NER include: extracting important named entities from legal, financial, and medical documents, classifying content for news providers, improving the search algorithms, and etc.

# Deep Learning Approach to NER
In this project, I used BERT(Bidirectional Encoder Representations from Transformers) which is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. You can find information about the BERT on [Google AI Blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html). 

![Figure 1: NER using BERT](https://www.vamvas.ch/wp-content/uploads/2019/06/bert-for-ner-768x661.png)

# Environment:
I run the code on Google cloud using n1-standard-8 (8 vCPUs, 30 GB memory). 

Main File: bert.py
command to run: nohup python3 bert.py > bert_logs.out &

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
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Lambda
from seqeval.metrics import classification_report
```

## Dataset Information
Here, we are going to extract the information of the biological based dataset
```python
data = pd.read_csv('train.csv', encoding = 'utf8')
data = data.drop(['POS'], axis =1)
data = data.fillna(method="ffill")
```

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

    def wordTag(self):
        words = set(list(self.data['Word'].values))
        words.add('PADword')
        tags = list(set(self.data["Tag"].values))
        return words, tags, len(words), len(tags)

    def find_largest_sentence(self):
        # Largest sentence Length
        largest_sen = max(len(sen) for sen in self.sentences)
        print('biggest sentence has {} words'.format(largest_sen))

    def plot_sentence_based_on_length(self):
        # Distribution of the length of the sentences
        plt.style.use("ggplot")
        plt.hist([len(sen) for sen in self.sentences], bins=50)
        plt.show()

getter = sentenceGetter(data)
sent = getter.get_next()
words, tags, n_words, n_tags = getter.wordTag()
sentences = getter.sentences
```
Total no. of words = 6363 and total no. of tags = 20

I created a class called sentenceGetter which is used to extract the words from the text and groupby them with their tags. A simple snippet of the result shown below
```
[('The', 'O'), ('absence', 'B-Quality'), ('of', 'O'), ('many', 'O'), ('other', 'O'), ('organisms', 'B-Biotic_Entity'), ('on', 'O'), ('the', 'O'), ('leaves', 'B-Biotic_Entity'), ('of', 'O'), ('planted', 'B-Eventuality'), ('trees', 'B-Biotic_Entity'), ('-', 'O'), ('except', 'O'), ('for', 'O'), ('the', 'O'), ('probably', 'O'), ('Asian', 'B-Biotic_Entity'), ('horse-chestnut', 'I-Biotic_Entity'), ('scale', 'I-Biotic_Entity'), ('insect', 'I-Biotic_Entity'), ('Pulvinaria', 'B-Biotic_Entity'), ('regalis', 'I-Biotic_Entity'), ('Canard', 'O'), ('the', 'O'), ('dothidiomycete', 'B-Biotic_Entity'), ('leaf', 'I-Biotic_Entity'), ('blotch', 'I-Biotic_Entity'), ('fungus', 'I-Biotic_Entity'), ('Guignardia', 'B-Biotic_Entity'), ('aesculi', 'I-Biotic_Entity'), ('(', 'O'), ('Peck', 'O'), (')', 'O'), ('V.B.', 'O'), ('Stewart', 'O'), ('of', 'O'), ('North', 'B-Location'), ('American', 'I-Location'), ('origin', 'B-Eventuality'), ('(', 'O'), ('Augustin', 'O'), ('2005', 'O'), (')', 'O'), ('and', 'O'), ('in', 'B-Time'), ('the', 'I-Time'), ('last', 'I-Time'), ('few', 'I-Time'), ('years', 'I-Time'), ('a', 'O'), ('powdery', 'B-Biotic_Entity'), ('mildew', 'I-Biotic_Entity'), ('Erysiphe', 'B-Biotic_Entity'), ('flexuosa', 'I-Biotic_Entity'), ('(', 'O'), ('Peck', 'O'), (')', 'O'), ('U.', 'O'), ('Braun', 'O'), ('et', 'O'), ('S.', 'O'), ('Takamatsuhas', 'O'), ('that', 'O'), ('is', 'O'), ('either', 'O'), ('invasive', 'B-Quality'), ('from', 'O'), ('North', 'B-Location'), ('America', 'I-Location'), ('or', 'O'), ('may', 'O'), ('be', 'O'), ('of', 'O'), ('Balkan', 'B-Location'), ('origin', 'B-Eventuality'), ('(', 'O'), ('Denchev', 'O'), ('2008', 'O'), (')', 'O'), ('-', 'O'), ('has', 'O'), ('given', 'O'), ('this', 'O'), ('moth', 'B-Biotic_Entity'), ('an', 'O'), ('almost', 'O'), ('free', 'B-Eventuality'), ('reign', 'I-Eventuality'), ('to', 'O'), ('colonise', 'B-Eventuality'), ('clean', 'B-Quality'), ('plants', 'B-Biotic_Entity'), ('.', 'O')]
```
So, Total number of the senteces comes out to be 2133 which is very less but quite challenging as well.

Then, I started trying to find out the longest sentence in the sentences for padding.

```python
# Largest sentence Length
largest_sen = getter.find_largest_sentence()
print('biggest sentence has {} words'.format(largest_sen))
```

The longest sentence has 122 words. Then, I plot a graph to find out what would be our distribution based on the length of the sentence.

```python
# Distribution of the length of the sentences
getter.plot_sentence_based_on_length()
```
A digram to show the distribution the words in the sentences
![Distribution of the words](https://github.com/ajaykedia1992/NameEntityRecognition/blob/master/SentencesRepresentation.png?raw=true)

Based on the distribution, we created the word length = 50 and convert them to their tags index

```python
max_len = 50
X = [[w[0]for w in s] for s in sentences]
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("PADword")
    new_X.append(new_seq)
    
tags2index = {t:i for i,t in enumerate(tags)}
y = [[tags2index[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tags2index["O"])
print(new_X[15])
print(y[15])
```
Snippet of the result
```python
['Higher', 'up', 'the', 'food', 'web', 'predatory', 'consumers', '—', 'especially', 'voracious', 'starfish', '—', 'eat', 'other', 'grazers', '(', 'e.g.', 'snails', ')', 'and', 'filter', 'feeders', '(', 'e.g.', 'mussels', ')', '.', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword', 'PADword']
[5, 5, 5, 1, 0, 10, 17, 5, 5, 9, 10, 5, 19, 5, 10, 5, 5, 10, 5, 5, 10, 17, 5, 5, 10, 5, 5]
```

## Splitting the dataset for training and testing
I used Sklearn library for splitting the training dataset 
```python
X_tr, X_te, y_tr, y_te = train_test_split(new_X, y, test_size=0.1, random_state=2018)
```

## Embedding the sentences
For this, I used ELMo for embedding the sentences. Tensorflow_hub provides a model to fetch ELMo
```python 
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
```

Running above block of code for the first time will take some time because ELMo is almost 400 MB. Next we use a function to convert our sentences to ELMo embeddings:

```python
batch_size = 32
sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x,    tf.string)),"sequence_len": tf.constant(batch_size*[max_len])
                     },
                      signature="tokens",
                      as_dict=True)["elmo"]
```

## Training
for this, we are going to use a batch size of 32. Now, let built our neural network
```python

input_text = Input(shape=(max_len,), dtype=tf.string)
embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
x = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)
model = Model(input_text, out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
```

since we have 32 as the batch size, feeding the network must be in chunks that are all multiples of 32:

```python
X_tr, X_val = X_tr[:50*batch_size], X_tr[-16*batch_size:]
y_tr, y_val = y_tr[:50*batch_size], y_tr[-16*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),batch_size=batch_size, epochs=3, verbose=1)
```
Running above block of code, our training will start and we will get the following snippet

```python
Train on 1600 samples, validate on 512 samples
Epoch 1/3
1600/1600 [==============================] - 428s 267ms/step - loss: 0.4899 - val_loss: 0.2399
Epoch 2/3
1600/1600 [==============================] - 418s 261ms/step - loss: 0.2104 - val_loss: 0.1733
Epoch 3/3
1600/1600 [==============================] - 417s 261ms/step - loss: 0.1573 - val_loss: 0.1440
```

## Testing
We also create the test datasets as such that it would be in the multiple of the batch size.
```python
X_te = X_te[:6 * batch_size]
test_pred = model.predict(np.array(X_te), verbose=1)
idx2tag = {i: w for w, i in tags2index.items()}
```

Following is the snippet of the result
```python
 32/192 [====>.........................] - ETA: 37s
 64/192 [=========>....................] - ETA: 25s
 96/192 [==============>...............] - ETA: 18s
128/192 [===================>..........] - ETA: 11s
160/192 [========================>.....] - ETA: 5s 
192/192 [==============================] - 34s 175ms/step
```

Now, accuracy will not tell me about the result. So, I started looking into the precision, recall and f-score of the prediction

```python
def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PADword", "O"))
        out.append(out_i)
    return out


def test2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p].replace("PADword", "O"))
        out.append(out_i)
    return out


pred_labels = pred2label(test_pred)
test_labels = test2label(y_te[:6 * 32])
print(classification_report(test_labels, pred_labels))
```

It gave me following result
```python
                                 precision    recall  f1-score   support

                  Biotic_Entity       0.78      0.87      0.82       425
                    Eventuality       0.77      0.81      0.79       410
                        Quality       0.69      0.41      0.52       146
                          Value       0.59      0.67      0.62        15
Aggregate_Biotic_Abiotic_Entity       0.71      0.74      0.73       133
                 Abiotic_Entity       0.74      0.88      0.80        57
                           Unit       1.00      0.67      0.80         9
                           Time       0.58      0.51      0.54        37
                       Location       0.86      0.60      0.71        10

                      micro avg       0.75      0.77      0.76      1242
                      macro avg       0.75      0.77      0.75      1242

```

## Result and Future Work
0.76 F1 score is an good achievement on a small dataset and based on 3 epochs but it can be further increased by increasing epochs and using different pre-training model, feature engineering, word embedding and many more. Also, 

# References
1. http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
2. https://jalammar.github.io/illustrated-bert/
3. https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede
4. https://www.vamvas.ch/bert-for-ner/
5. https://www.depends-on-the-definition.com/named-entity-recognition-with-residual-lstm-and-elmo/
6. https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html
7. https://github.com/google-research/bert









