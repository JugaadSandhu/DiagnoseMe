#!/usr/bin/python3
import random, sys

def split(data_filename, train_filename, test_filename):
    
    trainf = open(train_filename,"w")
    testf = open(test_filename,"w")
    first_line = True
    f = open(data_filename)
    for line in f:
        if first_line:
            trainf.write(line)
            testf.write(line)
            first_line = False
            continue
    
        if random.randrange(5)==0: testf.write(line)
        else: trainf.write(line)
    f.close()
    testf.close()
    trainf.close()

if __name__ == "__main__":
    if len(sys.argv)!=1:
        print('''
Usage: splitData.py 
    splits data.csv into train.csv and test.csv
''')
        sys.exit(1)


    split("data.csv", "train.csv", "test.csv")
