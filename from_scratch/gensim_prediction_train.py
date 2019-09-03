import gensim
import pymorphy2

import argparse

parser = argparse.ArgumentParser(description='Generating sentences v0.01')
parser.add_argument('word', metavar='word_to_predict', type=str, nargs='+',
                    help='an integer for the accumulator')
args = parser.parse_args()
predict_word = "мишка"
print("ARGS: %s" %(args))
if args.word:
    predict_word = args.word[0]

model_path = "G:\\New folder\\models\\gensim\\wordstat_100MB_model"
sequences = []
with open(file_path) as read_f:
    for line in read_f: 
        sequences.append(line.split(" "))
print(len(sequences))

model = gensim.models.Word2Vec(min_count=1)
model.build_vocab(sequences)
model.train(sequences, total_examples=model.corpus_count, epochs=model.iter)

model.save("G:\\New folder\\models\\gensim\\wordstat_100MB_model")

model.predict_output_word(['казань', "продажа"])