from ultralytics import YOLO

class v8(object):

	# init model
	def __init__(self, model_name):
		self.model = YOLO(model_name)
		self.conf = 0.5
