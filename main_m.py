import os
import sys

import argparse

#from timeit import default_timer
import yaml
import hashlib
import socket

import time

# ======== PLEASE MODIFY ========
# where is the repo
repoRoot = r'.'
# to CUDA\vX.Y\bin
os.environ['PATH'] = r'/usr/local/cuda/bin' + ';' + os.environ['PATH']

import numpy as np
import mxnet as mx

import cv2

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('config', type=str, nargs='?', default=None)
parser.add_argument('-g', '--gpu_device', type=str, default='', help='Specify gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None, 
	help='model checkpoint to load; by default, the latest one.'
	'You can use checkpoint:steps to load to a specific steps')
parser.add_argument('-n', '--network', type=str, default='MaskFlownet')
# inference resize for validation and prediction
parser.add_argument('--resize', type=str, default='')

args = parser.parse_args()
ctx = [mx.cpu()] if args.gpu_device == '' else [mx.gpu(gpu_id) for gpu_id in map(int, args.gpu_device.split(','))]
infer_resize = [int(s) for s in args.resize.split(',')] if args.resize else None

import network.config
# load network configuration
with open(os.path.join(repoRoot, 'network', 'config', args.config)) as f:
	config = network.config.Reader(yaml.load(f))

# create directories
def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
#mkdir('logs')
#mkdir(os.path.join('logs', 'val'))
#mkdir(os.path.join('logs', 'debug'))
#mkdir('weights')
mkdir('flows')

# find checkpoint
import path
prefix = args.checkpoint
log_file, run_id = path.find_log(prefix)	

checkpoint, steps = path.find_checkpoints(run_id)[-1]

# initiate
from network import get_pipeline
pipe = get_pipeline(args.network, ctx=ctx, config=config)

# load parameters from given checkpoint
print('Load Checkpoint {}'.format(checkpoint))
sys.stdout.flush()
network_class = getattr(config.network, 'class').get()
print('Load the weight for the network')
pipe.load(checkpoint)
if network_class == 'MaskFlownet':
	print('Fix the weight for the head network')
	pipe.fix_head()
sys.stdout.flush()

import predict_m
checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
start = time.time()
predict_m.predict(pipe, os.path.join(repoRoot, 'flows', checkpoint_name), batch_size=1, resize = infer_resize)
print(time.time() - start)
sys.exit(0)
