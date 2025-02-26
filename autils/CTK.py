
class CTK(object):
    def __init__(self, root):
        self.root = root
        self.root.title("图像识别程序")
        self.root.geometry("880x660")
        # self.image_frame = tk.Canvas(self.root, bg="gray", width=550, height=400)
        self.image_frame = tk.Canvas(self.root, bg="gray", width=590, height=440)
        self.image_frame.place(x=20, y=20)
        self.model_message = "图像显示"
        self.image_frame.create_text(275, 200, text="图像显示", font=("Helvetica", 12), tags="pre_display_none")
        custom_label = tk.Label(root, text="识别结果输出区", font=("Helvetica", 14))
        custom_label.place(x=650, y=20)

        self.result_frame = Canvas(root, bg="gray", width=200, height=300)
        self.result_frame.place(x=630, y=50)
        self.bordered_label = tk.Label(self.result_frame, text="识别状态程序输出", font=("Helvetica", 16), bg="gray",
                                       fg="black", relief="solid", bd=2)
        # self.result_frame.create_text(100, 20, text="识别程序状态输出", font=("Helvetica", 16))

        self.start_button = tk.Button(root, text="开始", command=self.start_camera, bg="orange", width=5, height=2,
                                      font=("Helvetica", 16))
        self.start_button.place(x=350, y=500)
        self.red_button = tk.Button(root, text="➔", command=self.kill_predict, bg="red", fg="black", width=5, height=2,
                                    font=("Helvetica", 24, "bold"))
        self.red_button.place(x=750, y=500)

        self.tip_text = tk.Label(root, text="灰色矩形框为可显示输出数据框，黄色矩形按钮表示开始按钮",
                                 font=("Helvetica", 10))
        self.tip_text.place(x=20, y=620)

        self.status_label = tk.Label(root, text="空闲", font=("Helvetica", 16), bg="gray", fg="black", width=16,
                                     height=2, relief="solid",
                                     bd=2)
        self.status_label.place(x=630, y=365)
        self.done_lock = True
        self.cap = None
        self.camera_thread = None
        self.is_running = False

        self.tags = []
        self.yolo_last_results = ["正在实时监测......"]
        self.init_img = cv2.imread("1.png")