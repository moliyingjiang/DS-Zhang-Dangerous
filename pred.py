from autils.v8 import v8
from autils.libcalculate import *

import threading
import pyttsx4


class Predict(object):
	"""docstring for Predict"""
	def __init__(self, model_path):
		#super(Predict, self).__init__()
		#self.arg = arg
		self.class_names_list = ["bucket", "hook", "person", "cement_mixer", "excavator", "crane"]
		
		self.YOLO = v8(model_path)
		self.model = self.YOLO.model

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
		metrics = self.model.predict(self.img)
		for m in metrics:
			positions_box = m.boxes.cpu()

			class_name = positions_box.cls.tolist()
			conf = positions_box.conf.tolist()
			xyxy = positions_box.xyxy.tolist()

			for index,x in enumerate(xyxy):
				if float(conf[index]) > self.YOLO.conf:
					# position
					a,b,c,d = int(x[0]), int(x[1]), int(x[2]), int(x[3])
					# data_info
					label_list = [self.class_names_list[int(class_name[index])], round(float(conf[index]),2),a,b,c,d]
					# 3d
					dis,angle,xyz = self.info_rs.Info_process(self.depth_intrin, self.aligned_depth_frame, label_list)
					self.label_array.append([self.class_names_list[int(class_name[index])], dis, angle, [a,b,c,d], xyz])
					c1, c2 = (a,b), (c,d)
			# print(class_name)
			# print(xyxy)
		self.frm = self.img
		# except
		if self.img is None or self.img.size == 0:
			print("Error: Frame is empty!")
			return None, self.color_image

		self.end_time = time.time()


		# define color
		rag_color = (0,255,0)

		for index,i in enumerate(self.label_array):
			print(i[0])
			if i[0] == "person":
				self.person_message_points.append([i[-2], i[-1]])
				self.del_index_list.append(index)
		'''end'''

		self.label_array = [item for idx, item in enumerate(self.label_array) if idx not in set(self.del_index_list)]

		'''cal'''
		for index_i,i in enumerate(self.person_message_points):
			for index_j,j in enumerate(self.label_array):
				# print(i[1])
				x_p, _, z_p = i[1]
				x_o, _, z_o = j[-1]
				dis_and = self.info_rs.dangerous_area(x_p, z_p, x_o, z_o)
				status = False
				if dis_and < 0.3:
					status = True
				self.person_message_points[index_i].append([j[0],dis_and, j[-2],status])
		'''cal end'''

		'''reg circle'''
		text = ""
		for index,i in enumerate(self.person_message_points):
			print(i[2:])
			rag_color = (0,255,0)
			x1,y1,x2,y2 = i[0]
			cv2.rectangle(self.frm,(x1,y1),(x2,y2),rag_color,2)
			cv2.putText(self.frm,"person",(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,rag_color,2)
			for index_o, o in enumerate(i):
				if index_o > 1:
					x1,y1,x2,y2 = o[-2]
					# if det.end_time-det.start_time
					rag2_color = (255,0,0)
					# cv2.rectangle(self.frm,(x1,y1),(x2,y2),rag2_color,2)
					# cv2.putText(self.frm,o[0],(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,rag2_color,2)
					if o[-1] and o[0] != "excavator" and o[0] != "crane":
						# print(o[-1])
						name = self.chinese_l(o[0])
						num = int(o[1] * 100)
						text = text + f"第{index+1}个人距离{name}有{num}厘米" + "\n"
						rag2_color = (0,0,255)
					# 不画吊车
					# if o[0] != "crane":
					cv2.rectangle(self.frm,(x1,y1),(x2,y2),rag2_color,2)
					cv2.putText(self.frm,o[0],(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,rag2_color,2)
		# 临时
		# ['crane', 0.162692962355475, [365, 282, 619, 423], True]
		# cv2.rectangle(self.frm,(365,282),(619,423),(0,255,0),2)
		# cv2.putText(self.frm,'crane',(365,282 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

		print("\n\n")

		text = text + "危险！请远离！"

		'''reg end'''

		'''text'''

		# if self.danger_count_status:
			# print("危险！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
		self.danger_count_status = False
		temp_list = []

		self.danger_distance = temp_list
		# print(self.danger_distance)
		# print(f'detect: {self.label_array}')
		if text != "危险！请远离！":
			return self.frm, self.color_image, text
		else:
			return self.frm, self.color_image, ""
