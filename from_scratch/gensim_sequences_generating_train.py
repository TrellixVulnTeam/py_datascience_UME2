import gensim
import pymorphy2
import datetime
import time
import os
import logging
file_path = "G:\\New folder\\subfile\\month-2011-12-qtraf_0_2"
model_save_path = "G:\\New folder\\models\\gensim\\phrases_45MB_model_clean"
train_log_path = "G:\\New folder\\logs\\gensim_train_log.txt"

#file_path = "/home/neuron/worstat_archives/wordstat_splitting/month-2011-12-qtraf_processed"
#train_log_path = "/home/neuron/logs/gensim_train_log.txt"
#model_save_path = "/home/neuron/models/gensim/wordstat_big_model"

log_file = open(train_log_path, "a")
log_file.write("Training Dataset :Name: %s :Size: %s :Date: %s \n" %(file_path, os.path.getsize(file_path), datetime.datetime.now()))

def train_model_on_dataset(model, dataset_path, update_vocabulary=False):
    sequences = []
    with open(dataset_path) as read_f:
        for line in read_f: 
            sequences.append(line.split(" "))
    print(len(sequences))
    start_time = time.time()
    print("Now train on %s \n" % (dataset_path))
    model.build_vocab(sequences, update=update_vocabulary)
    model.train(sequences, total_examples=model.corpus_count, epochs=model.iter)
    end_time = time.time() - start_time
    log_file.write("Elapsed time: %s secs. for file: %s \n" %(end_time))
    return model

def train_new_model_by_dataset(model, dataset_path, save_model_path, postfix):
    start_time = time.time()
    if postfix:
        for i in range(4):
            next_series_fn = dataset_path + postfix + str(i)
            if os.path.exists(next_series_fn):
                if i == 0:
                    model = train_model_on_dataset(model, next_series_fn, update_vocabulary=False)
                else:
                    model = train_model_on_dataset(model, next_series_fn, update_vocabulary=True)                    
            else:
                break
    print("Train is ended\n")
    model.save(save_model_path)
    print("Model saved on: %s \n" %(save_model_path))
    end_time = time.time() - start_time
    log_file.write("Elapsed time: %s" %(end_time))

model = gensim.models.Word2Vec(min_count=1)
train_new_model_by_dataset(model, file_path, save_model_path=model_save_path, postfix="_")

log_file.close()
model.predict_output_word(['казань', "покупка", "квартиры"])