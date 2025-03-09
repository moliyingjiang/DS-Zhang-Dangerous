# auther: xzy

from pred import *
from libdet import *
# from utils.CTK import *
from tkinter import Canvas
from PIL import Image, ImageTk
# from test import police
import tkinter as tk
import cv2
import threading
import time
import sys
import requests
import json

from tkinter import filedialog

class CameraApp:
    '''

    定义一些基础的变量以及TK图标库以及标签

    '''

    def __init__(self, root):
        self.root = root
        # self.police = police()
        self.root.title("危险区域人员闯入监测")
        self.root.geometry("880x660")
        # self.image_frame = tk.Canvas(self.root, bg="gray", width=550, height=400)
        self.image_frame = tk.Canvas(self.root, bg="gray", width=590, height=440)
        self.image_frame.place(x=20, y=20)
        self.model_message = "监控显示"
        self.image_frame.create_text(275, 200, text="监控显示", font=("Helvetica", 12), tags="pre_display_none")
        custom_label = tk.Label(root, text="人员闯入监测结果", font=("Helvetica", 14))
        custom_label.place(x=650, y=20)

        self.result_frame = Canvas(root, bg="gray", width=200, height=300)
        self.result_frame.place(x=630, y=50)
        self.bordered_label = tk.Label(self.result_frame, text="识别状态程序输出", font=("Helvetica", 16), bg="gray",
                                       fg="black", relief="solid", bd=2)
        # self.result_frame.create_text(100, 20, text="识别程序状态输出", font=("Helvetica", 16))

        self.file_button = tk.Button(root, text="选择文件", command=self.select_file, bg="lightyellow", width=10, height=2,
                                 font=("Helvetica", 12))
        self.file_button.place(x=25, y=500)
        self.start_button = tk.Button(root, text="开始", command=self.start_camera, bg="orange", width=5, height=2,
                                      font=("Helvetica", 16))
        self.start_button.place(x=350, y=500)
        self.start_button = tk.Button(root, text="继续", command=self.resume_camera, bg="white", width=5, height=2,
                                      font=("Helvetica", 12))
        self.start_button.place(x=180, y=500)
        self.red_button = tk.Button(root, text="➔Esc", command=self.kill_predict, bg="orange", width=5, height=2,
                                    font=("Helvetica", 16, "bold"))
        self.red_button.place(x=500, y=500)
        self.status_label2 = tk.Button(root, text="紧急报警", command=self.police, font=("Helvetica", 16), bg="white", fg="red", width=10,
                                     height=2, relief="solid",
                                     bd=2) 

        self.status_label2.place(x=743, y=600)

        self.tip_text = tk.Label(root, text="灰色矩形框为可显示输出数据框，黄色矩形按钮表示开始按钮  !! Warning: 选择文件后请点击继续按钮",
                                 font=("Helvetica", 10))
        self.tip_text.place(x=20, y=620)

        self.status_label = tk.Label(root, text="空闲", font=("Helvetica", 16), bg="gray", fg="black", width=16,
                                     height=2, relief="solid",
                                     bd=2)
        self.status_label.place(x=630, y=365)

        
        # 在这个界面的开始按钮的左侧加入一个选择文件的按钮栏，把选择的文件的地址放到self.bagpath
        
        self.done_lock = True
        self.cap = None
        self.camera_thread = None
        self.is_running = False

        self.tags = []
        self.mode = "0"
        self.yolo_last_results = ["正在实时监测......"]
        self.init_img = cv2.imread("1.png")

        self.start_time = time.time()

        self.v8_det() # 直接初始化

    def police(self):
        print("police")
        token = '243eb0062a18451c90ac6d123edf34e0' #在pushplus网站中可以找到
        # token = '你的token' #在pushpush网站中可以找到
        title= "Warning!" #改成你要的标题内容
        connect = ""
        if "正在实时监测" not in self.yolo_last_results[0]:
            content = self.yolo_last_results[0] + "请管理员通知误入人员离开！" #改成你要的正文内容
        else:
            content = self.yolo_last_results[0]
        url = 'http://www.pushplus.plus/send'
        data = {
            "token":token,
            "title":title,
            "content":content
        }
        body=json.dumps(data).encode(encoding='utf-8')
        headers = {'Content-Type':'application/json'}
        requests.post(url,data=body,headers=headers)


    def predict_text_display(self, predict_text):
        self.result_frame.delete("predict_text_display")
        self.result_frame.create_text(100, 16, text=predict_text, font=("Helvetica", 12), tags="predict_text_display")
        self.root.update_idletasks()  # Update display

    def select_file(self):
        self.is_running = False
        # Open a file dialog and store the selected file path in self.bagpath
        self.bagpath = filedialog.askopenfilename(title="选择文件", filetypes=[("All Files", "*.*")])
        if self.bagpath:
            print(f"Selected file: {self.bagpath}")
        self.det.info_rs.bagpath = self.bagpath
        self.det.info_rs.refresh()

    def resume_camera(self):
        self.is_running = True
            
    def start_camera(self):
        self.model_message = "模型加载中..."
        self.image_frame.create_text(275, 250, text=self.model_message, font=("Helvetica", 12),
                                     tags="pre_display_none")  # 创建并显示：模型加载中...
        self.root.update_idletasks()  # Update display 更新画面
        
        if not self.is_running:
            self.is_running = True
            # self.capture_loop()
            # 删除这里，线程运行的方式会导致画面闪动
            self.camera_thread = threading.Thread(target=self.capture_loop)
            self.camera_thread.start()

    def capture_loop(self):
        frame = self.init_img
        predict_text_arr = []
        predict_text_display_count = 40
        # device, half, model, names, colors = init()
        # 允许检测
        with open('output.txt', 'w', encoding='utf-8') as f:
            f.write("START")

        # self.v8_det()

        while True:
            # insert frame
            if not self.is_running:
                continue
            
            if self.done_lock:
                self.image_frame.delete("pre_display_none")  # 删除一开始DeBug用到的TK初始文字显示
                self.done_lock = False  # 模型加载完成只显示一次

                '''
                TK display
                '''
                # TK显示模型加载完毕
                self.tip_text = tk.Label(root, text="模型加载完毕！", font=("Helvetica", 10))
                self.tip_text.place(x=350, y=578)

                self.status_label.config(text="监测中")

            '''

            动态显示"模型加载中......."

            '''
            
            predict_text_display_count -= 1  # 保证动态区间控制的值在发生变化
            self.end_time = time.time()
            if ((self.end_time-self.start_time) % 15) < 0.1:
                self.yolo_last_results = ["正在实时监测......"]
                self.start_time = time.time()
            if predict_text_display_count % 4 == 0:  # 显示计数区间值隔帧动态--(根据自己的实际帧率调整，帧率高就调高点，帧率低就调低点)
                predict_text_arr.append(".")
                predict_text = f'AI模型监测中{"".join(predict_text_arr)}'  # 将列表转化为字符串方便显示
                if len(predict_text_arr) <= 6:  # 限制六个省略号
                    self.predict_text_display(predict_text)
            
            # 避免计数区间值的干涸与超值
            if predict_text_display_count == 0:
                predict_text_display_count = 40
                predict_text_arr.clear()

            ''''''
            # 调用V8进行检测
            self.deting()
            frame = self.init_img
            '''
            显示图像以及检测结果
            '''
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 避免图像蓝色调过重的问题/修复1

            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.image_frame.delete("image")  # Clear previous image
            self.image_frame.create_image(275, 200, image=img_tk, tags="image")  # Add new image
            self.result_frame.delete("done")  # Clear previous tag
            self.result_frame.delete("yolo_result_font")  # Clear previous tag

            self.root.update_idletasks()  # Update display

            '''

            将检测结果显示在输出框中

            '''
            pre_out_size = 120  # 设置初始的标签显示位置
            yolo_last_results = self.yolo_last_results # YOLO处理以后最后的结果
            for temp in yolo_last_results:
                # print(temp)
                # if temp != "":
                temp += "\n危险！请远离！"
                self.result_frame.create_text(100, pre_out_size, text=temp, font=("Helvetica", 9),
                                              tags="yolo_result_font")  # Add new tag to display
                pre_out_size += 38  # 自动换行间隔设定

            '''
            非常重要的地方，一定要更新画面

            '''
            self.root.update_idletasks()  # Update display

    def kill_predict(self):
        self.status_label.config(text="转动中")
        sys.exit()

    def pytt_start(self, text, engine):
        engine.say(text)
        engine.runAndWait()
        # engine.endLoop()   # 加入这一行
        engine.stop()

    def v8_det(self):
        ##### Voice
        engine = pyttsx4.init()
        engine.setProperty('voice', 'zh') #开启支持中文
        # engine.setProperty('voice',"zh")
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate-40)
        # 设置音量  范围为0.0-1.0  默认值为1.0
        engine.setProperty('volume', 0.7)

        self.engine = engine

        self.once_pyttsx_lock = True

        parser = argparse.ArgumentParser()
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        opt = parser.parse_args()

        ##### model
        model_path = "amodels/best.pt"
        if self.mode == "1":
            self.det = Predict(model_path)
        else:
            self.det = PreDict(opt)
            
    def deting(self):
        
        try:
                
            # self.det.bagpath
            frm, result_frame, text = self.det.predict()
            if result_frame is None:
                return None
            frm = np.array(frm)  # 转换为 NumPy 数组

            if (self.det.end_time-self.det.start_time) >= 18:
                    app.yolo_last_results = ["正在发现闯入人员......"]

            if (((self.det.end_time-self.det.start_time) >= 30) or (self.once_pyttsx_lock and (self.det.end_time-self.det.start_time) >= 1.5)) and text != "":
                # print("sssssssssssssssss")
                self.once_pyttsx_lock = False
                self.det.start_time = time.time()
                print(text)
                # pytt_start(text)
                app.yolo_last_results = [text]

                th = threading.Thread(target=self.pytt_start, args=(text,self.engine))
                th.start()
                # th.join()

            # frm = frm[..., ::-1]  # 从 RGB 转换为 BGR，OpenCV 使用 BGR

            # 如果图像是 GPU 上的 Tensor，迁移到 CPU 并转换为 NumPy 数组
            if isinstance(frm, torch.Tensor):
                frm = frm.cpu().numpy()
            app.init_img = result_frame

        except:
            print("Finding...")


if __name__ == '__main__':
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

