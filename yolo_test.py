import torch
import cv2
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import non_max_suppression
import time

model = attempt_load("runs/train/exp9/weights/best.pt")
model.to("cpu")
dataset = LoadImages("491.jpg")
for path, img, im0s, vid_cap in dataset:
	img = torch.from_numpy(img).to("cpu")
	img = img.float()  # uint8 to fp16/32
	img = img / 255.0  # 0 - 255 to 0.0 - 1.0
	if len(img.shape) == 3:
		img = img[None]  # expand for batch dim
	while True:
		start = time.time()
		pred = model(img)[0]
		conf_thres = .01
		pred = non_max_suppression(pred, conf_thres)
		print(time.time()-start)