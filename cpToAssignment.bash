#!/bin/bash
CWD=$(pwd)

cd ../../../a2/naiveBayes/
for i in diabetes  heart  heartBalanced  heartUnique
do
 rm -fr "$i"
 cp -r "$CWD/$i" .
done
