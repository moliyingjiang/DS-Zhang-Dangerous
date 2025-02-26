import torch.backends.cudnn as cudnn
from numpy import random
import torch
import cv2
from autils.libcalculate import *
import numpy as np
import argparse

# import pyttsx3
import pyttsx4
import time
import threading

import sys
import os

# 将 utils/libspare 路径添加到 sys.path
sys.path.append(os.path.abspath("autils/libspare"))
# print(sys.path)

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
	scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from models.experimental import attempt_load

class PreDict(object):
	"""docstring for Predict"""
	def __init__(self, opt):

		self.imgsz = 640
		weights = "amodels/best2.pt" # dir
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		half = self.device.type != 'cpu'  # half precision only supported on CUDA
		self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
		stride = int(self.model.stride.max())  # model stride
		# imgsz = check_img_size(imgsz, s=stride)  # check img_size
		print(self.imgsz)
		self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
		self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

		self.opt = opt
		self.info_rs = TARGET_INFO()

		self.danger_distance = []
		self.danger_count_status = True

		self.start_time = time.time()
		self.end_time = time.time()

		self.time_lock = False

	def frame(self):

		self.intr, self.depth_intrin, self.color_image, self.depth_image, self.aligned_depth_frame, self.pitch_rad = self.info_rs.get_aligned_images()  # 获取对齐的图像与相机内参
		'''如果没有图像存在 || 只要有一个不存在就会返回错误'''
		if not self.depth_image.any() or not self.color_image.any():
			print(f"Error('---------- Stream is not exist ----------------')") # 返回错误信息

		''' Apply colormap on depth image (image must be converted to 8-bit per pixel first) '''
		self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)

		''' 
		Stack both images horizontally 
		--------------------------------------
		注意：这一步必须得做！否则图像无法对齐导致坐标映射出现问题
		'''
		self.images = np.hstack((self.color_image, self.depth_colormap)) # 对齐深度图像以及RGB图像 || RGB图像传递给yolov5进行识别

		# 创建传递图像
		self.img = self.color_image

	def det_data(self):
		self.detect_end_status = False
		self.frm = None
		self.label_array = []
		self.person_xyz_array = []

		'''person'''
		self.person_message_points = []
		self.del_index_list = []

	def chinese_l(self, text):
		en_l = ["bucket", "hook", "person", "cement_mixer", "excavator", "crane"]
		chinese_l = ["铲子", "吊钩", "人", "水泥车", "挖掘机", "吊车"]
		index = en_l.index(text)
		return chinese_l[index]

	def predict(self):
		self.det_data() # refresh det data
		self.frame() # refresh frame
		img = self.img.copy() # copy img
		device = self.device # copy device
		model = self.model # copy model
		names = self.names # copy names
		# Run inference
		if self.device.type != 'cpu':
			self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
		old_img_w = old_img_h = self.imgsz
		old_img_b = 1

		# img = torch.from_numpy(img).to(device)
		img = np.array(img)
		img = img.astype(np.float32)  # 转换为 float32 类型
		img = torch.from_numpy(img).to(device)  # 转换为 tensor，并移动到设备上

		img = img.permute(2, 0, 1)  # 改变维度顺序：从 [H, W, C] 到 [C, H, W]

		img /= 255.0  # 归一化到 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# 打印调试信息，查看经过 unsqueeze 后的尺寸
		# print(f"Input image shape after unsqueeze: {img.shape}")

		# Warmup
		if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
			old_img_b = img.shape[0]
			old_img_h = img.shape[2]
			old_img_w = img.shape[3]
			for i in range(3):
				model(img, augment=self.opt.augment)[0]

		with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
			pred = model(img, augment=self.opt.augment)[0]

		# 一会儿处理
		pred = non_max_suppression(pred, 0.7, 0.45) # conf iou 

		# 一会儿处理
		frm = None
		label_array = []
		person_xyz_array = []
		for i, det in enumerate(pred):  # detections per image
			s, im0 = '', self.color_image
			# s = ""

			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

				# Write results
				for *xyxy, conf, cls in reversed(det):

					label = f'{names[int(cls)]} {conf:.2f}'
					x = xyxy

					# label_array.append(names[int(cls)])

					# print(f'num is {a} {b} {c} {d}')
					# if isinstance(im0, torch.Tensor):
						# im0 = im0.cpu().numpy()
					a,b,c,d = int(x[0]), int(x[1]), int(x[2]), int(x[3])
					label_list = [names[int(cls)], round(float(conf),2),a,b,c,d]

					# 电🗼线中心点 角度、距离
					# d_dis, d_ang = self.info_rs.Info_process(depth_intrin, aligned_depth_frame, ["crane", 0, 335,291, 100,100])
					# print(f"电🗼 线中心 距离及角度为 {d_dis} / {d_ang}")

					dis,angle,xyz = self.info_rs.Info_process(self.depth_intrin, self.aligned_depth_frame, label_list)

					# label_array.append(f'dis is {dis} =|= angle is {angle} !!! mac_dis is {mac_dis}')

					label_array.append([names[int(cls)], dis, angle, [a,b,c,d], xyz])
					# label_array.append("++++++++++")

					# label_array.append(f'angle is {angle}')
					c1, c2 = (a,b), (c,d)

				frm = im0
				if frm is None or frm.size == 0:
					print("Error: Frame is empty!")
					return None, self.color_image

				self.end_time = time.time()


		# define color
		rag_color = (0,255,0)

		'''person'''
		person_message_points = []
		del_index_list = []

		for index,i in enumerate(label_array):
			print(i[0])
			if i[0] == "person":
				person_message_points.append([i[-2], i[-1]])
				del_index_list.append(index)
		'''end'''

		label_array = [item for idx, item in enumerate(label_array) if idx not in set(del_index_list)]

		'''cal'''
		for index_i,i in enumerate(person_message_points):
			for index_j,j in enumerate(label_array):
				# print(i[1])
				x_p, _, z_p = i[1]
				x_o, _, z_o = j[-1]
				dis_and = self.info_rs.dangerous_area(x_p, z_p, x_o, z_o)
				status = False
				if dis_and < 0.3:
					status = True
				person_message_points[index_i].append([j[0],dis_and, j[-2],status])
		'''cal end'''

		'''reg circle'''
		text = ""
		for index,i in enumerate(person_message_points):
			print(i[2:])
			rag_color = (0,255,0)
			x1,y1,x2,y2 = i[0]
			cv2.rectangle(frm,(x1,y1),(x2,y2),rag_color,2)
			cv2.putText(frm,"person",(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,rag_color,2)
			for index_o, o in enumerate(i):
				if index_o > 1:
					x1,y1,x2,y2 = o[-2]
					# if det.end_time-det.start_time
					rag2_color = (255,0,0)
					# cv2.rectangle(frm,(x1,y1),(x2,y2),rag2_color,2)
					# cv2.putText(frm,o[0],(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,rag2_color,2)
					if o[-1] and o[0] != "excavator" and o[0] != "crane":
						# print(o[-1])
						name = self.chinese_l(o[0])
						num = int(o[1] * 100)
						text = text + f"第{index+1}个人距离{name}有{num}厘米" + "\n"
						rag2_color = (0,0,255)
					cv2.rectangle(frm,(x1,y1),(x2,y2),rag2_color,2)
					cv2.putText(frm,o[0],(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,rag2_color,2)

		print("\n\n")

		text = text + "危险！请远离！"

		'''reg end'''

		temp_list = []
		'''text'''
		self.danger_distance = temp_list

		# create a thing of class
		self.frm = frm

		if text != "危险！请远离！":
			return frm, self.color_image, text
		else:
			return frm, self.color_image, ""



# if __name__ == '__main__':

# 	##### Voice
# 	engine = pyttsx4.init()
# 	engine.setProperty('voice', 'zh') #开启支持中文
# 	# engine.setProperty('voice',"zh")
# 	rate = engine.getProperty('rate')
# 	engine.setProperty('rate', rate-40)
# 	# 设置音量  范围为0.0-1.0  默认值为1.0
# 	engine.setProperty('volume', 0.7)

# 	def pytt_start(text):
# 		engine.say(text)
# 		engine.runAndWait()
# 		# engine.endLoop()   # 加入这一行
# 		engine.stop()

# 	once_pyttsx_lock = True

# 	##### model
# 	model_path = "best.pt"
# 	det = Predict(model_path)
# 	while True:
# 		frm, result_frame, text = det.predict()
# 		frm = np.array(frm)  # 转换为 NumPy 数组

# 		if (((det.end_time-det.start_time) >= 30) or (once_pyttsx_lock and (det.end_time-det.start_time) >= 1.5)) and text != "":
# 			# print("sssssssssssssssss")
# 			once_pyttsx_lock = False
# 			det.start_time = time.time()
# 			print(text)
# 			# pytt_start(text)
			
# 			th = threading.Thread(target=pytt_start, args=(text,))
# 			th.start()
# 			# th.join()
		
# 		# frm = frm[..., ::-1]  # 从 RGB 转换为 BGR，OpenCV 使用 BGR

# 		# 如果图像是 GPU 上的 Tensor，迁移到 CPU 并转换为 NumPy 数组
# 		if isinstance(frm, torch.Tensor):
# 			frm = frm.cpu().numpy()
# 		cv2.imshow("result", result_frame)
# 		cv2.waitKey(1)

# 	cv2.destroyAllWindows()





