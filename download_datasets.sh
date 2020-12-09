#!/bin/bash

declare -a arr=("ml-1m" "ml-latest-small" "ml-25m")

# now loop through the above array
for i in "${arr[@]}"
do
   wget "http://files.grouplens.org/datasets/movielens/$i.zip"
   unzip "$i.zip" && rm "$i.zip"
done