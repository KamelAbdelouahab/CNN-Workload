#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import io
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
from PIL import Image

def readCNN(model_file,proto_file):
	net = caffe.Net(proto_file,1,weights=model_file)
	net.forward()
	return net
def plotDataTransfer(net):
	conv_layer_bandwidth = []
	conv_layer_name = []
	for l in net._layer_names:
		layer_id = list(net._layer_names).index(l)
		layer_type =  net.layers[layer_id].type
		if (layer_type == 'Input'):
			conv_layer_name.append(l)
			conv_layer_bandwidth.append(net.blobs[l].data.size/1000000)
		if (layer_type == 'Convolution'):
			conv_layer_name.append(l)
			conv_layer_bandwidth.append(net.blobs[l].data.size/1000000)
		if (layer_type == 'Pooling'):
			conv_layer_name.append(l)
			conv_layer_bandwidth.append(net.blobs[l].data.size/1000000)
	i = 0;
	for l in conv_layer_name:
		print("layer %s, %2.2f MB/img" %(l,conv_layer_bandwidth[i]));
		i += 1

	pos = np.arange(len(conv_layer_bandwidth))
	fig = plt.figure()
	# plt.rc('text', usetex=True)
	# plt.rc('font', family='sans-serif')
	plt.bar(pos,conv_layer_bandwidth,align='center', alpha=0.5)
	plt.xticks(pos,conv_layer_name)
	plt.ylabel(r'Data Transfer (MB/s)')
	plt.title(r'Bandwidth Requirements')
	plt.show()
	#fig.savefig('./alexnet_bandwidth.pdf', bbox_inches='tight')


if __name__ == '__main__':
	if (len(sys.argv) == 3):
		os.environ['GLOG_minloglevel'] = '2'     # Supresses Display on console
		import caffe;
		proto_file = sys.argv[1]
		model_file = sys.argv[2]
		net = readCNN(model_file,proto_file)
		plotDataTransfer(net)
else:
		print("Not enought arguments")
		print("python Bandwidth.py <path_to_proto> <path_to_caffemodel>")
