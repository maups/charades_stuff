# Description

Training files for action classification using the Charades Activity Challenge dataset.

# Instructions

Clone this repository:
```
$ git clone https://github.com/maups/charades_stuff/
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
