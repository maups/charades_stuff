# Description

Training files for scenario classification using the Charades Activity Challenge dataset.

# Instructions for training using rgb frames

Clone this repository:
```
$ git clone -b scenario --single-branch https://github.com/maups/charades_stuff/
$ cd charades_stuff/
```
Download Charades' annotations and rgb frames:
```
$ wget http://vuchallenge.org/vu17_charades.zip
$ unzip vu17_charades.zip
$ rm vu17_charades.zip
$ wget http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar
$ tar -xf Charades_v1_rgb.tar
$ rm Charades_v1_rgb.tar
```
Normalize videos to 128x128 frame size and 1fps:
``` 
$ ./create_scaled_frames.sh
```
Create helper files:
```
$ g++ -std=c++11 create_helper_files.cpp
$ mkdir helper_files
$ ./a.out
$ rm a.out
```
Run the training for a CNN:
```
$ python cnn.py
```
