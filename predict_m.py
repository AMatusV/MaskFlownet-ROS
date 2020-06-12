import os
from reader import kitti_m
import cv2
import numpy as np

# PLEASE MODIFY the paths specified in sintel.py and kitti.py

def predict(pipe, prefix, batch_size = 8, resize = None):

	#kitti_resize = (512, 1152) if resize is None else resize
	kitti_resize = resize
	kitti_dataset = kitti_m.read_dataset_testing(resize = kitti_resize)
	prefix = prefix + '_kitti'
	if not os.path.exists(prefix):
		os.mkdir(prefix)

	for k, dataset in kitti_dataset.items():
		#output_folder = os.path.join(prefix, k)
		#if not os.path.exists(output_folder):
		#	os.mkdir(output_folder)
		output_folder = prefix

		img1 = kitti_dataset[k]['image_0']
		img2 = kitti_dataset[k]['image_1']
		cnt = 0
		for result in pipe.predict(img1, img2, batch_size = 1, resize = kitti_resize):
			flow, occ_mask, warped = result
			out_name = os.path.join(output_folder, '%06d_10.png' % cnt)
			cnt = cnt + 1

			pred = np.ones((flow.shape[0], flow.shape[1], 3)).astype(np.uint16)
			pred[:, :, 2] = (64.0 * (flow[:, :, 0] + 512)).astype(np.uint16)
			pred[:, :, 1] = (64.0 * (flow[:, :, 1] + 512)).astype(np.uint16)
			cv2.imwrite(out_name, pred)
			