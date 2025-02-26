#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Robot -> librealinfo.py
@IDE    ：Sublime Text
@Author ：XuZhiyuan
@Date   ：2024/03/18 18:24
@Desc   ：Realsense / Getting info of target
=================================================='''


# HOW TO STOP STRAMING? Example: TARGET_INFO().pipeline.stop()


'''
ROBOT || USE REALSENSE
-------------------------
2024.03.18

INIT INFO OF REALSENSE || Getting Depth_View | Color_View | Align_View

requirment : pip install pyrealsense2
'''


import cv2
import time

import pyrealsense2 as rs
import numpy as np

import torch
import math
import random



'''
INFORMATION OF REALSENSE
----------------------------
2024.3.18 INIT CAM || INIT STREAM

Depth_view || RGB_view - = 这两部分需要保证大小相同并且图像对齐
'''

class TARGET_INFO: # 目标信息 || 来源于Realsense

    '''INIT CAM==Realsense || pyrealsense2'''
    def __init__(self):
        
        '''----------------- 相机参数配置 START-----------------'''
        '''基本定义'''
        self.pipeline = rs.pipeline()  # 定义流程pipeline
        self.config = rs.config()  # 定义配置config

        ''' Frame weight&height '''
        # self.frame_weight = 848
        self.frame_weight = 640
        self.frame_height = 480

        self.accel_status = True
        
        '''DEPTH || COLOR''' # 需要设备检测取消注释以下内容
        # self.config.enable_stream(rs.stream.depth, self.frame_weight, self.frame_height, rs.format.z16, 30) # made a attempt || FPS 30 is MAX
        # self.config.enable_stream(rs.stream.color, self.frame_weight, self.frame_height, rs.format.bgr8, 30) # made a attempt || FPS 30 is MAX
        # if self.accel_status:
        #     self.config.enable_stream(rs.stream.accel,rs.format.motion_xyz32f,63)
        #     self.config.enable_stream(rs.stream.gyro,rs.format.motion_xyz32f,200)  # 获取 IMU 数据（加速度计）

        # 读取录制好的视频
        self.config.enable_device_from_file("../input.bag", repeat_playback=False)

        '''STAEAM/FREAM START'''
        self.profile = self.pipeline.start(self.config)  # 流程开始
        # self.profile.get_device().hardware_reset()
        
        self.align_to = rs.stream.color  # 与color流对齐self
        self.align = rs.align(self.align_to)
        '''---------------- 相机参数配置 ENDING -------------------'''



    ''' 获得已 排列/处理 后的所有图像 '''
    def get_aligned_images(self):

        accel_status=self.accel_status
        
        ''' 获取相关图像帧 '''
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        
        pitch_rad = "None"
        # 若测俯仰角
        if accel_status:
            accel_frame = frames.first_or_default(rs.stream.accel)  # 获取加速度数据
            
            if not accel_frame:
                pass

            # 加速度数据
            # accel_data = accel_frame.as_motion_frame()
            # accel_x = accel_data.get_motion_data().x
            # accel_y = accel_data.get_motion_data().y
            # accel_z = accel_data.get_motion_data().z

            # # 使用公式：pitch = atan(ax / sqrt(ay^2 + az^2))
            # pitch_rad = math.atan2(accel_x, math.sqrt(accel_y ** 2 + accel_z ** 2)) # 俯仰角 弧度
            # print("pitch rad is {pitch_rad}")

        aligned_frames = self.align.process(frames)  # 获取对齐帧

        '''
        将对齐帧中的深度帧与RGB帧进行分离
        ---------------
        '''
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        '''
        ------------------ 相机参数的获取 --------------------
        '''
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        

        '''
        camera_parameters = 
        {   
            'fx': intr.fx, 'fy': intr.fy,
            'ppx': intr.ppx, 'ppy': intr.ppy,
            'height': intr.height, 'width': intr.width,
            'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
        }
        '''

        # # 保存内参到本地
        # with open('./intrinsics.json', 'w') as fp:
        # json.dump(camera_parameters, fp)

        '''------------------ ENDING --------------------------'''

        '''十六位与八位深度图'''
        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）

        ''' 三通道深度图与三通道RGB图 '''
        depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
        color_image = np.asanyarray(color_frame.get_data())  # RGB图

        ''' 
        返回 {相机内参、深度参数、彩色图、深度图、齐帧中的depth帧} 
        '''
        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame, pitch_rad


    # 采样，取平均
    def sample_points(self, x1, y1, x2, y2, num_points=10):
        # 确保坐标是有效的
        assert x1 < x2 and y1 < y2, "x1, y1 应该分别小于 x2, y2"
        
        # 存储随机点的列表
        points = []
        
        # 随机采样点
        for _ in range(num_points):
            x = random.uniform(x1, x2)  # 在 x1 和 x2 之间生成随机浮点数
            y = random.uniform(y1, y2)  # 在 y1 和 y2 之间生成随机浮点数
            points.append([x, y])
        
        return points

    # 人距离危险区域
    def dangerous_area(self, x1, z1, x2, z2):
        person_and_dangerous_distance = math.sqrt((x1-x2) ** 2 + (z1-z2) ** 2)
        return person_and_dangerous_distance
    
    # 测距
    def Info_process(self, depth_intrin, aligned_depth_frame, frame_object_info):

        '''
        提取 类别名称 || 类别置信度 || 坐标xyxy
        '''
        class_name, confidence = frame_object_info[:2] # class / confidence
        x1, y1, x2, y2 = frame_object_info[2:] # xyxy of obj

        '''
        计算像素坐标系xyxy
        '''
        real_distance_list = real_angle_list = []
        sample_point_list = self.sample_points(x1, y1, x2, y2)
        for i in sample_point_list:
            ux = int(i[0])
            uy = int(i[1])
            dis = aligned_depth_frame.get_distance(ux, uy) # 在已经对齐的深度图像上进行相对距离估算
            self.camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [ux, uy], dis)  # 计算相机坐标系的xyz
            self.camera_xyz = np.round(np.array(self.camera_xyz), 3)  # 转成3位小数
            self.camera_xyz = self.camera_xyz.tolist() # 进一步处理
            if self.camera_xyz[2] != 0:
                real_distance_list.append(self.camera_xyz[2] + 0)
                # x,y,z = self.camera_xyz
                # real_angle_list.append(self.calculate_angle(x, y, z))
        if len(real_distance_list) > 0:
            real_distance = sum(real_distance_list) / len(real_distance_list) # 相机距离物体的距离
        else:
            real_distance = 0.0
        # real_angle = sum(real_angle_list) / len(real_angle_list)
        
        ux = int((x1 + x2)/2)  # 计算像素坐标系的x
        uy = int((y1 + y2)/2)  # 计算像素坐标系的y
        
        # if class_name == "crane":
        #     ux = int(x1) + 20  # 计算像素坐标系的x
        #     uy = int(y1) + 25  # 计算像素坐标系的y
        # if class_name == "excavator":
        #     ux = int(x2) - 15
        #     uy = int(y2) - 55
        # if class_name == "cement_mixer":
        #     ux = 303
        #     uy = 285

        '''---------------------------- 估算距离以及角度 start ------------------------------'''

        dis = aligned_depth_frame.get_distance(ux, uy) # 在已经对齐的深度图像上进行相对距离估算
        self.camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [ux, uy], dis)  # 计算相机坐标系的xyz
        
        self.camera_xyz = np.round(np.array(self.camera_xyz), 3)  # 转成3位小数
        self.camera_xyz = self.camera_xyz.tolist() # 进一步处理
        # # print(f"fffffffffffff{self.camera_xyz}")

        # '''
        # 以下加的是误差
        # ------------------------
        # *** + xx || xx为误差
        # '''
        # real_distance = self.camera_xyz[2] + 0 # 相机距离物体的距离
        x,y,z = self.camera_xyz
        real_angle = self.calculate_angle(x, y, z)
        # real_angle = self.camera_xyz[0] + 0 # 相机与物体之间的相对角度 # 错误

        # '''----------------------------------- ENDING ---------------------------------'''
        self.distance = real_distance
        self.angle = real_angle

        return real_distance, real_angle, (x, y, z) # 返回相机与物体的相对距离以及相对角度

    def calculate_angle(self, x1, y1, z1, x2=-0.037, y2=0.088, z2=1.172):
        # [-0.031, 0.091, 1.159] 🗼
        # 计算两个向量的点积
        # if x1 == 0.0:
        #     return 0.0
        dot_product = x1 * x2 + y1 * y2 + z1 * z2

        # 计算两个向量的模长
        v1_magnitude = math.sqrt(x1**2 + y1**2 + z1**2)
        v2_magnitude = math.sqrt(x2**2 + y2**2 + z2**2)

        if v2_magnitude == 0 or v1_magnitude == 0:
            return 0.0

        # 计算余弦值
        cos_theta = dot_product / (v1_magnitude * v2_magnitude)

        # 限制 cos_theta 在 [-1, 1] 范围内，避免计算时因为浮动导致错误
        cos_theta = max(-1, min(1, cos_theta))

        # 计算夹角的弧度值
        angle_rad = math.acos(cos_theta)

        # 转换为度数
        angle_deg = math.degrees(angle_rad)

        return angle_rad

    def real_dis(self, distance, angle, height=0.25):

        '''
        HEIGHT ** 2 + REALDISTANCE ** 2 == DISTANCE ** 2
        '''
        # height = 0.28 # 相机距离地面的真实高度 || 后续要根据自身的步态 【匍匐、正常站立、上楼梯】的真实高度进行设定和调整
        if angle < 0:
            angle = - angle
        
        # req angle
        # angle = angle * 180 / 3.1415926

        mac_distance = distance * np.tan(angle)

        '''勾股计算 || 已知斜边和一直角边求另一直角边'''
        # real_distance = math.sqrt(abs(distance**2 - height**2))

        return mac_distance


if __name__ == '__main__':
    
    pass



