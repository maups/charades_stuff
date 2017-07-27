#!/bin/bash

cat $1 |grep TRAIN |cut -d " " -f 2- > train.dat
cat $1 |grep VAL |cut -d " " -f 2- > val.dat
gnuplot show.gp
display show.png
