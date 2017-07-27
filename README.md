# Description

Training files for action classification using the Charades Activity Challenge dataset.

# Instructions for training using flow only

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
![Results for lstm.py](results/res_lstm.png?raw=true "lstm.py")
Run the training for a 2-layer LSTM network:
```
$ python stacked_lstm.py
```
![Results for stacked_lstm.py](results/res_stacked_lstm.png?raw=true "stacked_lstm.py")

# Instructions for training using flow + texture

Download Charades' texture features:
```
$ wget http://ai2-website.s3.amazonaws.com/data/Charades_v1_features_rgb.tar.gz
$ tar -xzf Charades_v1_features_rgb.tar.gz
$ rm Charades_v1_features_rgb.tar.gz
```
Run the training for a 3-layer fully connected network:
```
$ python fc_rgb.py
```
![Results for fc_rgb.py](results/res_fc_rgb.png?raw=true "fc_rgb.py")
Run the training for a 2-layer LSTM network:
```
$ python stacked_lstm_rgb.py
```
![Results for stacked_lstm_rgb.py](results/res_stacked_rgb.png?raw=true "stacked_lstm_rgb.py")
