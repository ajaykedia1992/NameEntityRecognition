# importing files
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


data = pd.read_csv('train.csv', encoding='utf8')
data = data.drop(['POS'], axis=1)
data = data.fillna(method="ffill")

# calling sentenceGetter class and get_next function
getter = sentenceGetter(data)
sent = getter.get_next()
words, tags, n_words, n_tags = getter.wordTag()
sentences = getter.sentences
largest_sen = getter.find_largest_sentence()
getter.plot_sentence_based_on_length()

# The longest sentence has 140 words in it and we can see that almost all of the sentences have less than 60 words in them.

# Based on the distribution, we will create the length word = 50 and padded with the word 'padding'.

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
new_X[15]


tags2index = {t:i for i,t in enumerate(tags)}
y = [[tags2index[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tags2index["O"])
print(y[15])


X_tr, X_te, y_tr, y_te = train_test_split(new_X, y, test_size=0.1, random_state=2018)
sess = tf.Session()
K.set_session(sess)
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


batch_size = 32
def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x,    tf.string)),"sequence_len": tf.constant(batch_size*[max_len])
                     },
                      signature="tokens",
                      as_dict=True)["elmo"]

# Creating neural network
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


X_tr, X_val = X_tr[:50*batch_size], X_tr[-16*batch_size:]
y_tr, y_val = y_tr[:50*batch_size], y_tr[-16*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),batch_size=batch_size, epochs=3, verbose=1)

X_te = X_te[:6 * batch_size]
test_pred = model.predict(np.array(X_te), verbose=1)
idx2tag = {i: w for w, i in tags2index.items()}


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
