import os
import sys
import argparse
import yaml
#import hashlib
#import socket
import numpy as np
import mxnet as mx
import cv2
import time

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class maskflownet:

	def __init__(self, argv):
		# MaskFlownet init with args
		repoRoot = r'.'
		os.environ['PATH'] = r'/usr/local/cuda/bin' + ';' + os.environ['PATH']

		parser = argparse.ArgumentParser(add_help=False)
		parser.add_argument('config', type=str, nargs='?', default=None)
		parser.add_argument('-g', '--gpu_device', type=str, default='', help='Specify gpu device(s)')
		parser.add_argument('-c', '--checkpoint', type=str, default=None, 
			help='model checkpoint to load; by default, the latest one.'
			'You can use checkpoint:steps to load to a specific steps')
		parser.add_argument('-n', '--network', type=str, default='MaskFlownet')
		parser.add_argument('--resize', type=str, default='')
		args = parser.parse_args(argv[1:])  # argv[1:] to ignore the 1st element (scipt's name)

		ctx = [mx.cpu()] if args.gpu_device == '' else [mx.gpu(gpu_id) for gpu_id in map(int, args.gpu_device.split(','))]
		self.infer_resize = [int(s) for s in args.resize.split(',')] if args.resize else None

		# load network configuration
		import network.config
		with open(os.path.join(repoRoot, 'network', 'config', args.config)) as f:
			config = network.config.Reader(yaml.load(f))

		# find checkpoint
		import path
		prefix = args.checkpoint
		log_file, run_id = path.find_log(prefix)
		checkpoint, steps = path.find_checkpoints(run_id)[-1]

		# initiate
		from network import get_pipeline
		self.pipe = get_pipeline(args.network, ctx=ctx, config=config)

		# load parameters from given checkpoint
		print('Load Checkpoint {}'.format(checkpoint))
		sys.stdout.flush()
		network_class = getattr(config.network, 'class').get()
		print('Load the weight for the network')
		self.pipe.load(checkpoint)
		if network_class == 'MaskFlownet':
			print('Fix the weight for the head network')
			self.pipe.fix_head()
		sys.stdout.flush()

		# ROS topics
		self.cameraTopic = "usb_cam/image_raw"	
		self.flowTopic = "flow_topic"
		self.camera_sub = rospy.Subscriber(self.cameraTopic, Image, self.camera_cb)
		self.flow_pub = rospy.Publisher(self.flowTopic, Image, queue_size=1)
		self.image_msg = Image()

		self.bridge = CvBridge()
		self.prevCvImage = None
		self.currCvImage_msg = None
		self.imageRcvd_fl = False
		self.img1 = []
		self.img2 = []

	def camera_cb(self, data):
		self.image_msg = data
		self.imageRcvd_fl = True
		
	def predict(self):
		try:
			self.prevCvImage = self.currCvImage
			self.currCvImage = self.bridge.imgmsg_to_cv2(self.image_msg, "bgr8")

			self.img1[0] = self.prevCvImage
			self.img2[0] = self.currCvImage
			for result in self.pipe.predict(self.img1, self.img2, batch_size = 1, resize = self.infer_resize):
				flow, occ_mask, warped = result

				pred = np.ones((flow.shape[0], flow.shape[1], 3)).astype(np.uint16)
				pred[:, :, 2] = (64.0 * (flow[:, :, 0] + 512)).astype(np.uint16)
				pred[:, :, 1] = (64.0 * (flow[:, :, 1] + 512)).astype(np.uint16)
			
			self.flow_pub.publish(self.bridge.cv2_to_imgmsg(pred, "bgr16"))
		except CvBridgeError as e:
			print(e)

	def node(self):
		# ROS related code
		rospy.init_node("maskflownet_node", anonymous=True)

		rospy.loginfo("[MFN] Waiting for the first image")
		self.image_msg = rospy.wait_for_message(self.cameraTopic, Image)
		try:
			self.currCvImage = self.bridge.imgmsg_to_cv2(self.image_msg, "bgr8")
			rospy.loginfo("[MFN] First image received")
		except CvBridgeError as e:
			print(e)
		
		self.img1.append(self.currCvImage)
		self.img2.append(self.currCvImage)

		rate = rospy.Rate(30)
		while not rospy.is_shutdown():
			if self.imageRcvd_fl:
				#start = time.time()
				self.predict()
				#print(time.time() - start)

			rate.sleep()

def main(argv):
	mfn = maskflownet(argv)
	try:
		mfn.node()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	main(sys.argv)	