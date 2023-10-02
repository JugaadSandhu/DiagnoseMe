#!/bin/bash
CWD=$(pwd)
for d in diabetes heart heartBalanced heartUnique
do
    cd "$d"
    # ../splitData.py # if you run this you will get different results each time
    ../naiveBayes.py
    tail -n 1 "out.csv"
    cd "$CWD"
done
