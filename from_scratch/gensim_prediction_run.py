import gensim
import pymorphy2

import argparse

parser = argparse.ArgumentParser(description='Generating sentences using gensim v0.01')
parser.add_argument('word', metavar='words_sentence', type=str, nargs='+',
                    help='an integer for the accumulator')
args = parser.parse_args()
# this sentence uses to generante various sentences
starting_sentence = []
# there will be generated sentences
generated_sentences = []
# print("ARGS: %s" %(args))
if args.word:
    starting_sentence = args.word

model_path = "G:\\New folder\\models\\gensim\\wordstat_100MB_model"

model = gensim.models.Word2Vec.load(model_path)
next_word_variants = model.predict_output_word(starting_sentence)
for ans in next_word_variants:
    new_sen = starting_sentence[:]
    new_sen.append(ans[0])
    print(" ".join(new_sen))
