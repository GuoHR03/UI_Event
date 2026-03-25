import numpy as np
import time
import os
import gc  # 引入垃圾回收
from PyQt6.QtCore import QThread, pyqtSignal

# Metavision SDK 导入
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

class CameraThread(QThread):
    # 信号：发送图像数据 (ndarray) 到 UI
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, palette_str="Dark"):
        super().__init__()
        self.device = None
        self.is_running = True
        self.is_recording = False
        
        # 算法对象占位符
        self.event_frame_gen = None 
        self.width = 640
        self.height = 480
        
        # 1. 初始化颜色参数
        self.palette = self._get_palette_enum(palette_str)
        
        # 2. 线程安全切换标志位
        self.pending_palette = None 
        self.update_flag = False 

    def _get_palette_enum(self, str_val):
        """辅助函数：将字符串转换为 SDK 枚举"""
        if str_val == "Light": return ColorPalette.Light
        elif str_val == "Gray": return ColorPalette.Gray
        elif str_val == "CoolWarm": return ColorPalette.CoolWarm
        else: return ColorPalette.Dark

    # 👇 外部调用此函数来修改颜色 (安全方法)
    def set_palette(self, palette_str):
        new_palette = self._get_palette_enum(palette_str)
        # 只有颜色变了才举旗
        if new_palette != self.palette:
            self.pending_palette = new_palette
            self.update_flag = True # 🚩 举旗：通知 run 循环在下一帧切换

    def on_cd_frame_cb(self, ts, cd_frame):
        """回调函数：处理生成好的帧"""
        if self.is_running:
            # 🔧 修复灰度图报错：如果是 2维 (Gray)，转为 3维 (BGR)
            if cd_frame.ndim == 2:
                cd_frame = np.stack((cd_frame,)*3, axis=-1)
            
            # 发送信号
            self.image_signal.emit(cd_frame)

    def run(self):
        try:
            # 1. 打开设备
            self.device = initiate_device("")
            
            # 2. 获取迭代器和分辨率
            mv_iterator = EventsIterator.from_device(device=self.device)
            self.height, self.width = mv_iterator.get_size()

            # 3. 初始创建算法对象
            self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=self.width, 
                sensor_height=self.height, 
                fps=25,
                palette=self.palette
            )
            # 绑定回调
            self.event_frame_gen.set_output_callback(self.on_cd_frame_cb)

            # 4. 开始循环处理事件
            for evs in mv_iterator:
                if not self.is_running:
                    break

                # 👇👇👇 安全切换逻辑 (在子线程内部执行) 👇👇👇
                if self.update_flag:
                    print(f"应用新颜色模式: {self.pending_palette}")
                    
                    # 更新颜色变量
                    self.palette = self.pending_palette
                    
                    # 销毁旧对象，创建新对象
                    self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                        sensor_width=self.width, 
                        sensor_height=self.height, 
                        fps=25, 
                        palette=self.palette
                    )
                    
                    # 【至关重要】新对象必须重新绑定回调函数！
                    self.event_frame_gen.set_output_callback(self.on_cd_frame_cb)
                    
                    # 放下旗子
                    self.update_flag = False
                # 👆👆👆 切换结束 👆👆👆

                # 处理事件
                self.event_frame_gen.process_events(evs)

        except Exception as e:
            print(f"CameraThread Error: {e}")
        
        finally:
            print("正在清理相机资源...")
            # 1. 停止录制
            if self.is_recording:
                self.stop_recording()
            
            # 2. 清理引用
            self.event_frame_gen = None
            
            # 3. 停止流并销毁设备
            if self.device:
                try:
                    stream = self.device.get_i_events_stream()
                    if stream: stream.stop()
                except RuntimeError: pass # 忽略 frame size mismatch
                except: pass
                
                del self.device
                self.device = None
            
            # 4. 强制垃圾回收
            gc.collect()
            print("相机资源已释放")

    def stop(self):
        self.is_running = False
        self.wait() 

    def start_recording(self):
        if self.device and not self.is_recording:
            filename = "recording_" + time.strftime("%y%m%d_%H%M%S", time.localtime()) + ".raw"
            log_path = os.path.join(os.getcwd(), filename)
            try:
                self.device.get_i_events_stream().log_raw_data(log_path)
                self.is_recording = True
                print(f"开始录制: {log_path}")
            except: pass

    def stop_recording(self):
        if self.device and self.is_recording:
            try: self.device.get_i_events_stream().stop_log_raw_data()
            except: pass
            self.is_recording = False
            print("停止录制")