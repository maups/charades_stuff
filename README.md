# Description

Training files for action classification using the Charades Activity Challenge dataset.

# Instructions for training

Clone this repository:
```
$ git clone https://github.com/maups/charades_stuff/
$ cd charades_stuff/
```
Download Charades' annotations and flow features:
```
$ wget http://vuchallenge.org/vu17_charades.zip
$ unzip vu17_charades.zip
$ rm vu17_charades.zip
$ wget http://ai2-website.s3.amazonaws.com/data/Charades_v1_features_flow.tar.gz
$ tar -xzf Charades_v1_features_flow.tar.gz
$ rm Charades_v1_features_flow.tar.gz
```
Create helper files:
```
$ g++ -std=c++11 create_helper_files.cpp
$ mkdir helper_files
$ ./a.out
$ rm a.out
```
Run the training for a 3-layer fully connected network:
```
$ python fc.py
```
![Results for fc.py](results/res_fc.png?raw=true "fc.py")
Run the training for a LSTM network:
```
$ python lstm.py
```
Run the training for a 2-layer LSTM network:
```
$ python stacked_lstm.py
```
