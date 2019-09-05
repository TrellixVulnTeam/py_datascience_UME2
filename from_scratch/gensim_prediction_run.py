import gensim
import pymorphy2
import argparse
import time
import datetime
import os



#Returns Top N words, that similars with
def getSimilarsForWord(word, model, top=10):
    parsed = morph.parse(word)
    try:
        pos = cotags[parsed[0].tag.POS]
    except KeyError:
        return [word]
    gensim_find_word = word + "_" + pos
    most_similars = model.most_similar([gensim_find_word], topn=top)
    return_list = []
    for sim in most_similars:
        sim_parsed = sim[0].split("_")
        if sim_parsed[1] == pos:
            return_list.append(sim_parsed[0])
    return return_list


#log_path = "/home/neuron/logs/gensim_run_log.txt"
log_path = "G:\\New folder\\logs\\gensim_run_log.txt"

log_file = open(log_path, "a")
#Time measurement
start_time = time.time()
parser = argparse.ArgumentParser(description='Generating sentences using gensim v0.02')
parser.add_argument('word', metavar='words_sentence', type=str, nargs='+',
                    help='Starting sentence to predict next')
args = parser.parse_args()
# this sentence uses to generante various sentences
starting_sentence = []
# there will be generated sentences
generated_sentences = []
# print("ARGS: %s" %(args))
if args.word:
    starting_sentence = args.word

model_path = "G:\\New folder\\models\\gensim\\wordstat_100MB_model"
#model_path = "/home/neuron/models/gensim/wordstat_big_model10"
model = gensim.models.Word2Vec.load(model_path)
next_word_variants = model.predict_output_word(starting_sentence, topn=20)


log_file.write("Session Date: %s Time: %.0f Model Size: %s \n" %(datetime.datetime.now(), time.time() - start_time, os.path.getsize(model_path)))
log_file.write("Input: %s\n" %(starting_sentence))

log_file.write("Output:\n")
for ans in next_word_variants:
    new_sen = starting_sentence[:]
    new_sen.append(ans[0])
    print(" ".join(new_sen))
    log_file.write("".join(new_sen) + "\n")
log_file.write("//---------------\n")
