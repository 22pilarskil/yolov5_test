# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import sys
import time
from pathlib import Path
import serial_test

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pyrealsense2 as rs
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync


@torch.no_grad()
def run(weights='yolov5s.pt',  # model
	source='0',  # directory for images
	project='runs/detect',  # directory for save results
	imgsz=640,  # inference size (pixels)
	conf_thres=0.25,  # confidence threshold
	iou_thres=0.45,  # NMS IOU threshold
	max_det=100,  # maximum detections per image
	name='exp',  # save results to project/name
	):

	# Initialize
	set_logging()
	device = select_device('')

	model = attempt_load(weights, map_location=device)  # load FP32 model

	names = model.module.names if hasattr(model, 'module') else model.names  # get class names

	view_img = check_imshow()
	cudnn.benchmark = True  # set True to speed up constant image size inference
	dataset = LoadStreams(source, img_size=imgsz, stride=int(model.stride.max()), auto=True)
	bs = len(dataset)  # batch_size

	vid_path, vid_writer = [None] * bs, [None] * bs

	# Run inference
	#if device.type != 'cpu':
	#	model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # #run once
	comm = serial_test.Coms()
	try:
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
		config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
		pipeline.start(config)
		while True:
			frames = pipeline.wait_for_frames()
			depth = frames.get_depth_frame()
			if not depth: continue
			color_frame = frames.get_color_frame()
			if not color_frame: continue
			img = np.array(color_frame.get_data())
			img = np.moveaxis(img, 2, 0) 
			print(img.shape)
			img = torch.from_numpy(img).to("cpu")
			img = img.float()  # uint8 to fp16/32
			img = img / 255.0  # 0 - 255 to 0.0 - 1.0
			if len(img.shape) == 3:
				img = img[None]  # expand for batch dim
			start = time.time()
			pred = model(img)[0]
			conf_thres = .01
			pred = non_max_suppression(pred, conf_thres)
			print(time.time()-start)
		       # Process predictions
			for i, det in enumerate(pred):  # detections per image
				s, im0 = f'{i}: ', imgc


				s += '%gx%g ' % img.shape[2:]  # print string
				annotator = Annotator(im0, line_width=3, pil=not ascii)
				if len(det):
					# Rescale boxes from img_size to im0 size
					det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

					# Print results
					for c in det[:, -1].unique():
				    		n = (det[:, -1] == c).sum()  # detections per class
				    		s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

				# Write results
				for *xyxy, conf, cls in reversed(det):
					c = int(cls)  # integer class
					label = f'{names[c]} {conf:.2f}'
					print('(' + str(int(xyxy[0])) + ',' + str(int(xyxy[1])) + ') - (' + str(int(xyxy[2])) + ',' + str(int(xyxy[3])) + ')')
					comm.send("header", str((xyxy[0] + xyxy[2]) / 2) + str((xyxy[0] + xyxy[2]) / 2))
					annotator.box_label(xyxy, label, color=colors(c, True))


			    # Stream results
			im0 = annotator.result()
			cv2.imshow(im0)
			cv2.waitKey(1)  # 1 millisecond
	except KeyboardInterrupt:
		pipeline.stop()

def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars())


if __name__ == "__main__":
    main()
