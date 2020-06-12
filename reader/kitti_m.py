import os
import cv2
import numpy as np

# ======== PLEASE MODIFY ========
kitti_root = r'/home/esteban/Downloads/kitti_samples'

def read_dataset_testing(path = None, editions = 'mixed', resize = None, samples = None):
	if path is None:
		path = kitti_root

	dataset = dict()
	dataset['2012'] = dict()
	dataset['2012']['image_0'] = []
	dataset['2012']['image_1'] = []

	#num_files = (len(os.listdir(path)) - 1) // 2
	num_files = len(os.listdir(path)) // 2
	if samples is not None:
		num_files = min(num_files, samples)
	for k in range(num_files):
		img0 = cv2.imread(os.path.join(path, '%06d_10.png' % k))
		img1 = cv2.imread(os.path.join(path, '%06d_11.png' % k))
		if resize is not None:
			img0 = cv2.resize(img0, resize)
			img1 = cv2.resize(img1, resize)
		dataset['2012']['image_0'].append(img0)
		dataset['2012']['image_1'].append(img1)

	return dataset

if __name__ == '__main__':
	#dataset = read_dataset(resize = (1024, 436))
	dataset = read_dataset_testing(resize = (1024, 436))
	# print(dataset['occ'][0].shape)
