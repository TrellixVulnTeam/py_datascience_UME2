import gensim
import pymorphy2
import argparse
import time
import datetime
import os



#Returns Top N words, that similars with
def getSimilarsForSentence(sentence, model, top=10):
    cur_sentences = [[el[0]] for el in model.wv.most_similar([sentence[0]], topn=top)]
    # #now we go through the sentence length
    news = sentence[1:]
    print("nwes: %s" %(news))
    for word in news:
        new_cur_sentences = []
        most_similars = model.wv.most_similar([word], topn=top)
        for sim_item in most_similars:
            sub_cur_sentences = []
            for cur_item in cur_sentences:
                #adding slice from previous element
                sub_cur_sentences.append(cur_item[:])
            for cur_item in sub_cur_sentences:
                cur_item.append(sim_item[0])
                new_cur_sentences.append(cur_item[:])
        cur_sentences = new_cur_sentences
    return cur_sentences


log_path = "/home/neuron/logs/gensim_run_log.txt"
#log_path = "G:\\New folder\\logs\\gensim_run_log.txt"

log_file = open(log_path, "a")
#Time measurement
start_time = time.time()
parser = argparse.ArgumentParser(description='Generating sentences using gensim v0.02')
parser.add_argument('sentence', metavar='words_sentence', type=str, nargs='+',
                    help='Starting sentence to predict next')
args = parser.parse_args()
# this sentence uses to generante various sentences
starting_sentence = []
# there will be generated sentences
generated_sentences = []
# print("ARGS: %s" %(args))
if args.sentence:
    starting_sentence = args.sentence

print(starting_sentence)
#model_path = "G:\\New folder\\models\\gensim\\wordstat_100MB_model"
model_path = "/home/neuron/models/gensim/wordstat_big_model10"
model = gensim.models.Word2Vec.load(model_path)

log_file.write("Session Date: %s Time: %.0f Model Size: %s \n" %(datetime.datetime.now(), time.time() - start_time, os.path.getsize(model_path)))
log_file.write("Input: %s\n" %(starting_sentence))

log_file.write("Output:\n")

variant_sentences = getSimilarsForSentence(starting_sentence, model, 3)


for sentence in variant_sentences:
    next_word_variants = model.predict_output_word(starting_sentence, topn=20)
    for ans in next_word_variants:
        new_sen = sentence[:]
        new_sen.append(ans[0])
        print(" ".join(new_sen))
        log_file.write("".join(new_sen) + "\n")
log_file.write("//---------------\n")
