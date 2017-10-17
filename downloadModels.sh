#!/bin/bash

wget --show-progress -P 'nets/alexnet.caffemodel' 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
wget --show-progress -P 'nets/googlenet.caffemodel' 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
wget --show-progress -P 'nets/vgg16.caffemodel/' 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
wget --show-progress -P 'nets/vgg19.caffemodel/' 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'

echo "Please Download ResNet Models at https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777"
