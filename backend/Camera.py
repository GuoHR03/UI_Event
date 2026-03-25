import time
import queue
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

class CameraThread(QThread):
    image_signal = pyqtSignal(object)
    finished_signal = pyqtSignal()

    def __init__(self, palette_type="Dark",fps=30,nn_interval_ms=20,target_queue=None,file_path = "C:/Users/tianmu/Downloads/pedestrians.raw"): #target_queue
        super().__init__()
        self.is_running = True
        self.is_recording = False
        self.target_queue = target_queue  # 预测线程的输入管道
        self.analysis_enabled = True      # 预测开关
        self.input_path = file_path
        delta_t_us = int(nn_interval_ms * 1000)

        # 1. 初始化相机
        try:
            self.device = initiate_device("")
            self.mv_iterator = EventsIterator.from_device(device=self.device, delta_t=delta_t_us)
        except:
            self.device = None
            self.mv_iterator = EventsIterator(input_path=self.input_path, delta_t=delta_t_us)
        self.width, self.height = self.mv_iterator.get_size()
        #显示效果
        palette_map = {
            "Dark": ColorPalette.Dark,
            "Light": ColorPalette.Light,
            "CoolWarm": ColorPalette.CoolWarm,
            "Gray": ColorPalette.Gray
        }
        palette = palette_map.get(palette_type, ColorPalette.Dark)

        # 设置帧率 保证大于0   
        if fps <= 0:
            fps = 30
        
        # 3. 初始化帧生成算法
        self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=self.width, sensor_height=self.height, fps=fps, palette=palette)
        self.event_frame_gen.set_output_callback(self._on_cd_frame_cb)

        # 在线程刚建好时，往队列里塞一个特别的“配置包” 传相机的参数
        if self.target_queue is not None:
            config_payload = {
                "msg_type": "CONFIG",
                "width": self.width,
                "height": self.height
            }
            
            while not self.target_queue.empty():
                try:
                    self.target_queue.get_nowait()
                except queue.Empty:
                    break
                    
            try:
                self.target_queue.put_nowait(config_payload)
            except queue.Full:
                print("无法传入相机参数")
    def _on_cd_frame_cb(self, ts, cd_frame):
        """算法生成图像后的回调"""
        if self.is_running:
            self.image_signal.emit(cd_frame.copy())

    def start_recording(self):
        i_events_stream = self.device.get_i_events_stream()
        if i_events_stream:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.raw"
            i_events_stream.log_raw_data(filename)
            self.is_recording = True
            print(f"开始录制原始数据: {filename}")

    def stop_recording(self):
        """停止 RAW 录制信号"""
        i_events_stream = self.device.get_i_events_stream()
        if i_events_stream:
            i_events_stream.stop_log_raw_data()
            self.is_recording = False
            print("停止录制并保存文件")

    def run(self):
        for evs in self.mv_iterator:
            if not self.is_running:
                break

            # UI显示
            self.event_frame_gen.process_events(evs)

            # 实时处理
            if self.analysis_enabled and self.target_queue is not None:
                if evs.size > 0:
                    try:
                        clean_array = np.column_stack((
                            evs['x'], 
                            evs['y'], 
                            evs['t'], 
                            evs['p']
                        )).astype(np.float32)
                        if self.target_queue.full():
                            self.target_queue.get_nowait()
                        self.target_queue.put_nowait(clean_array)
                    except queue.Full:
                        pass
            time.sleep(0.01)

        if self.is_running:
                    self.finished_signal.emit()

    def stop(self):
        self.is_running = False
        if self.is_recording:
            self.is_recording = False
            self.stop_recording()