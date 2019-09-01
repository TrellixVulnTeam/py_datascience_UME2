import os
import time
import gensim
import pymorphy2
import numpy as np
import tensorflow as tf
import unidecode
from keras_preprocessing.text import Tokenizer
 
tf.enable_eager_execution()
 
file_path = "/home/neuron/dataset/small_linux.txt"
file_path = "G:\\New folder\\month-2011-12-qtraf_small"

load_word2vec_path = "/home/neuron/dataset/model.bin"
load_word2vec_path = "G:\\New folder\\ruwikiruscorpora_tokens_elmo_1024_2019\\ruwikiruscorpora_upos_skipgram_300_2_2019\\model.bin"

#Now we load 
model = gensim.models.KeyedVectors.load_word2vec_format(load_word2vec_path, binary=True)
model.init_sims(replace=True)
morph = pymorphy2.MorphAnalyzer()
cotags = {
    'ADJF':'ADJ', # pymorphy2: word2vec 
    'ADJS' : 'ADJ', 
    'ADVB' : 'ADV', 
    'COMP' : 'ADV', 
    'GRND' : 'VERB', 
    'INFN' : 'VERB', 
    'NOUN' : 'NOUN', 
    'PRED' : 'ADV', 
    'PRTF' : 'ADJ', 
    'PRTS' : 'VERB', 
    'VERB' : 'VERB'
}

text = open(file_path).read()
 
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
 
encoded = tokenizer.texts_to_sequences([text])[0]
 
vocab_size = len(tokenizer.word_index) + 1
 
word2idx = tokenizer.word_index
idx2word = tokenizer.index_word

sequences = list()

for i in range(1, len(encoded)):
    sequence = encoded[i - 1:i + 1]
    sequences.append(sequence)
sequences = np.array(sequences)
#print(word2idx)
X, Y = sequences[:, 0], sequences[:, 1]
X = np.expand_dims(X, 1)
Y = np.expand_dims(Y, 1)

BUFFER_SIZE = 100
BATCH_SIZE = 100
dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(Model, self).__init__()
        self.units = units
        self.batch_size = batch_size
 
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
 
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_activation='sigmoid',
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
 
    def call(self, inputs, hidden):
        inputs = self.embedding(inputs)
        #print(inputs)
        output, states = self.gru(inputs, initial_state=hidden)
 
        output = tf.reshape(output, (-1, output.shape[2]))
 
        x = self.fc(output)
 
        return x, states
 
#This function returns only similar words that contains in train dataset
def sortSimilarListByDataset(words_list):
    ret_list = []
    for word in words_list:
        try:
            if word2idx[word]:
                ret_list.append(word)
        except KeyError:
            continue
    return ret_list
#Returns Top N words, that similars with
def getSimilarsForWord(word, top=10):
    parsed = morph.parse(word)
    pos = cotags[parsed[0].tag.POS]
    gensim_find_word = word + "_" + pos
    most_similars = model.most_similar([gensim_find_word], topn=top)
    return_list = []
    for sim in most_similars:
        sim_parsed = sim[0].split("_")
        if sim_parsed[1] == pos:
            return_list.append(sim_parsed[0])
    return return_list

 
embedding_dim = 100
 
units = 2048
 
keras_model = Model(vocab_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()
 
#checkpoint_dir = '.\\training_checkpoints_wordstat'
checkpoint_dir = '.\\training_checkpoints_wordstat_small2048'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=keras_model)

def loss_function(labels, logits):
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

EPOCHS = 10
# for epoch in range(EPOCHS):
#     start = time.time()
 
#     hidden = keras_model.reset_states()
 
#     for (batch, (input, target)) in enumerate(dataset):
#         with tf.GradientTape() as tape:
#             predictions, hidden = keras_model(input, hidden)
 
#             target = tf.reshape(target, (-1,))
#             loss = loss_function(target, predictions)
 
#             grads = tape.gradient(loss, keras_model.variables)
#             optimizer.apply_gradients(zip(grads, keras_model.variables))
 
#             if batch % 100 == 0:
#                 print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss))
 
#     if (epoch + 1) % 10 == 0:
#         checkpoint.save(file_prefix=checkpoint_prefix)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
current_word = "мишка"
input_eval = [word2idx[current_word]]
input_eval = tf.expand_dims(input_eval, 0)

#print("UNITS: %s" %(units))
hidden = [tf.zeros((1, units))]

#Now we find similars for start word
similar_words = getSimilarsForWord(current_word, 10)
similar_words.append(current_word)
dataset_words_list = sortSimilarListByDataset(similar_words)
#print("dataset_words_list %s" %(dataset_words_list))

sequences_lists = [[word] for word in dataset_words_list]
print(sequences_lists)
for sequence in sequences_lists:
    for i in range(4):
        input_eval = [word2idx[sequence[i]]]
        input_eval = tf.expand_dims(input_eval, 0)    

        predictions, hidden = keras_model(input_eval, hidden)
#         print("PREDICTIONS")
#         print(predictions)

        predicted_id = tf.argmax(predictions[-1]).numpy()

        sequence.append(idx2word[predicted_id])
        
for sequence in sequences_lists:
    print(" ".join(sequence))