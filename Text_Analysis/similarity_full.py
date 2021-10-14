import time
import pandas as pd
import csv
import sys
import numpy as np
from bert_serving.client import BertClient
from termcolor import colored

prefix_q = ''
topk = 1

def print_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

def run(sentences2encode, sentences2compare, fout):
    print_time()
    with open(sentences2encode) as fp:
        sentences = fp.readlines()
        sentences = [x.strip() for x in sentences] 
        print('%d sentences loaded, avg. len of %d' % (len(sentences), np.mean([len(d.split()) for d in sentences])))

    with open(sentences2compare) as fq:
        sentences2 = fq.readlines()
        sentences2 = [x.strip() for x in sentences2]
        print('%d sentences loaded, avg. len of %d' % (len(sentences2), np.mean([len(d.split()) for d in sentences2])))

    with BertClient(port=8190, port_out=5556) as bc:
        mat1 = bc.encode(sentences)
        mat2 = bc.encode(sentences2)
        res = np.dot(mat1,mat2.transpose())
        df = pd.DataFrame(data=res.astype(float))
        df.to_csv(fout, sep=',', header=False, float_format='%.4f', index=False)

    print_time()

if __name__== "__main__":
    list=['comp1100','comp1110','comp1600','comp1710','comp1730','comp2100','comp2120',
          'comp2310','comp2400','comp2410','comp2420','comp2550','comp2560','comp2610',
          'comp2620','comp2700','comp3120','comp3300','comp3310','comp3320','comp3425',
          'comp3430','comp3530','comp3600','comp3620','comp3701','comp3702','comp3703',
          'comp3704','comp3900','comp4300','comp4330','comp4610','comp4620','comp4670',
          'comp4691','comp4880']

    for i in range (len(list)):
        for j in range (len(list)):
            run("/home/yixincheng/8755project/WebAndReligion/raw_foundational_books/"+list[i]+'.txt',
                "/home/yixincheng/8755project/WebAndReligion/raw_foundational_books/"+list[j]+'.txt',
                "full/"+list[i][:8]+"_"+list[j][:8]+".csv")

