#!/bin/bash

declare -a arr=("ml-latest-small" "ml-1m" "ml-latest")

# now loop through the above array
for i in "${arr[@]}"
do
   wget "http://files.grouplens.org/datasets/movielens/$i.zip"
   unzip "$i.zip" && rm "$i.zip"
   mv "$i" data/
done