import pandas as pd
import numpy as np

#Version = 0.02
#wordstat_path = "G:\\New folder\\month-2011-12-qtraf"
#my_dataset_path = "G:\\New folder\\month-2011-12-qtraf_0_2"

#linux_wordstat_path = "/home/neuron/worstat_archives/wordstat_splitting/month-2011-12-qtraf"
#linux_save_path = linux_wordstat_path + "_processed"

def write_by_lines():
    counter = 0
    with open(my_dataset_path) as write_f:
        with open(wordstat_path) as read_f:
            for line in read_f:
                print(line)
                write_f.write(line)
                counter +=1
                if counter > 1000000:
                    print('Writing is ended!')
                    break

def wordstat_processing(read_path, save_path, read_bytes_count = 2000, seeks_count=10000, intervals=5000, start_value = 18000000):
    counter = start_value
    with open(save_path, "a") as write_f:
        with open(read_path, encoding="windows-1251") as read_f:
            read_f.seek(start_value)
            while True:
                #reduce intervals count
                intervals -= 1
                #reduce seek by "read_bytes_count"
                counter += read_bytes_count
                read_f.seek(counter)
                c = read_f.read(read_bytes_count)
                if (not c) or (intervals == 0):
                    break
                counter += seeks_count
                splited_c = c.split('\n')
                if counter > start_value:
                    for line in splited_c:
                        tab_split = line.split('\t')
                        if len(tab_split) == 2:
                            write_f.write(tab_split[1]+'\n')

def wordstat_processed_print(read_bytes_count = 1024, seeks_count=1024, intervals=1, start_value = 18000000):
    counter = start_value
    with open(my_dataset_path, "a") as write_f:
        with open(wordstat_path) as read_f:
            read_f.seek(start_value)
            while True:
                #reduce intervals count
                intervals -= 1
                #reduce seek by "read_bytes_count"
                counter += read_bytes_count
                read_f.seek(counter)
                c = read_f.read(read_bytes_count)
                if (not c) or (intervals == 0):
                    break                
                counter += seeks_count
                splited_c = c.split('\n')
                if counter > start_value:
                    for line in splited_c:
                        tab_split = line.split('\t')
                        if len(tab_split) == 2:
                            print(tab_split[1])

def divide_big_file_to_series(file_name = "", lines_count = 1000000):
    lines_counter = 0
    current_little_file_postfix = 0
    write_f_name = file_name + "_" + str(current_little_file_postfix)
    write_f = open(write_f_name, "a")
    print("Big file dividing started\n")
    with open(file_name, encoding="windows-1251") as open_f:
        #create new file
        for line in open_f:
            if lines_counter > lines_count:
                write_f.close()
                print("File %s done with %s lines\n" %(write_f_name, str(lines_counter)))
                current_little_file_postfix += 1
                write_f_name = file_name + "_" + str(current_little_file_postfix)
                write_f = open(write_f_name, "a")
                lines_counter = 0
            write_f.write(line)
            lines_counter += 1
        write_f.close()
        print("Job for file '%s' is Done\n" %(file_name))

wordstat_processing(linux_wordstat_path, linux_save_path, read_bytes_count = 1024*2000, seeks_count=1024*10000, intervals=3000, start_value = 16000000*1024)
#divide_big_file_to_series(linux_wordstat_path, 4000000)
