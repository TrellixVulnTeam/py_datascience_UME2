import pandas as pd
import numpy as np

#That file uses to generate files with less 
wordstat_path = "G:\\New folder\\month-2011-12-qtraf"
my_dataset_path = "G:\\New folder\\month-2011-12-qtraf_million"
def write_by_lines():
    counter = 0
    with open(my_dataset_path) as write_f:
        with open(wordstat_path) as read_f:
            for line in read_f.readline():
                print(line)
                write_f.write(line)
                counter +=1
                if counter > 1000000:
                    print('Writing is ended!')
                    break


def write_by_read():
    with open(wordstat_path) as read_f:
        while True:
            c = read_f.read(1024)
            splited_c = c.split('\n')
            if not c:
                break
            #print(repr(c))
            print(splited_c)

write_by_read()