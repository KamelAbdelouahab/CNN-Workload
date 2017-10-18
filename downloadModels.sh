#!/bin/bash

wget --show-progress -P 'nets/alexnet.caffemodel' 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
wget --show-progress -P 'nets/googlenet.caffemodel' 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
wget --show-progress -P 'nets/vgg16.caffemodel/' 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
wget --show-progress -P 'nets/vgg19.caffemodel/' 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'

echo "Please Download ResNet Models at: https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777"
echp "Downloadig ResNet models from DREAM-IP repo, sorry Microsoft ..."
wget --no-check-certificate  --show-progress -P 'nets/resnet50.caffemodel/' 'https://193.54.50.147/f/d67db1f32a/?raw=1'
wget --no-check-certificate  --show-progress -P 'nets/resnet101.caffemodel/' 'https://193.54.50.147/f/09ae3e9bf4/?raw=1'
wget --no-check-certificate  --show-progress -P 'nets/resnet152.caffemodel/' 'https://193.54.50.147/f/23fcaba38c/?raw=1'
wget --no-check-certificate  --show-progress -P 'nets/squeezenet.caffemodel/' 'https://193.54.50.147/f/52ed700be9/?raw=1'
