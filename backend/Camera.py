import time
import queue
import cv2
import numpy as np
import dv_processing as dv  
import h5py
from PyQt6.QtCore import QThread, pyqtSignal
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

class CameraThread(QThread):
    image_signal = pyqtSignal(object)
    finished_signal = pyqtSignal()

    def __init__(self, palette_type="Dark", fps=30, nn_interval_ms=20, target_queue=None, file_path=""): 
        super().__init__()
        self.is_running = True
        self.is_recording = False
        self.target_queue = target_queue  # 预测线程的输入管道
        self.analysis_enabled = True      # 预测开关
        self.input_path = file_path
        self.fps = fps if fps > 0 else 30
        delta_t_us = int(nn_interval_ms * 1000)

        # ==========================================
        # 1. 引擎分流：判断扩展名
        # ==========================================
        self.is_aedat4 = self.input_path and self.input_path.lower().endswith(".aedat4")
        self.is_h5 = self.input_path and self.input_path.lower().endswith(('.h5', '.hdf5'))

        if self.is_aedat4:
            # ---------------- 【AEDAT4 引擎初始化】 ----------------
            self.device = None  # aedat4 回放模式不支持 raw 录制
            self.dv_reader = dv.io.MonoCameraRecording(self.input_path)
            
            try:
                res = self.dv_reader.getEventResolution()
                self.width, self.height = res.width, res.height
            except:
                self.width, self.height = 640, 480
                
            self.dv_visualizer = dv.visualization.EventVisualizer((self.width, self.height))
            
        if self.is_aedat4:
            # ---------------- 【模仿彩色 Metavision 官方风格调色板】 ----------------
            if palette_type == "Light":
                # 背景白，正极黑，负极白(隐形)
                self.dv_visualizer.setBackgroundColor((255.0, 255.0, 255.0)) 
                self.dv_visualizer.setPositiveColor((0.0, 0.0, 0.0))       
                self.dv_visualizer.setNegativeColor((255.0, 255.0, 255.0))       
            elif palette_type == "Gray":
                # 背景灰，正极白，负极黑
                self.dv_visualizer.setBackgroundColor((128.0, 128.0, 128.0)) 
                self.dv_visualizer.setPositiveColor((255.0, 255.0, 255.0))   
                self.dv_visualizer.setNegativeColor((0.0, 0.0, 0.0))         
            elif palette_type == "CoolWarm":
                # 保留彩色冷暖色：背景暗灰，正极暖橙红，负极冷深蓝
                self.dv_visualizer.setBackgroundColor((30.0, 30.0, 30.0))    
                self.dv_visualizer.setPositiveColor((0.0, 128.0, 255.0))     
                self.dv_visualizer.setNegativeColor((255.0, 100.0, 0.0))     
            else: # Dark / 默认
                # 【关键修复】从 Ground Truth 中精确采样的红蓝彩色版
                # 背景色：采样为 #1C2433 (深海军灰)
                self.dv_visualizer.setBackgroundColor((28.0, 36.0, 51.0)) 
                # 正极事件：纯白 (White)
                self.dv_visualizer.setPositiveColor((255.0, 255.0, 255.0))   
                # 负极事件：采样为 #0066CC (深蓝色)
                self.dv_visualizer.setNegativeColor((0.0, 102.0, 204.0))

        elif self.is_h5:
            # ---------------- 【HDF5 引擎初始化】 ----------------
            self.device = None
            self.h5_file = h5py.File(self.input_path, 'r')
            self.events_dataset = self.h5_file['events']
            self.h5_dtypes = self.events_dataset.dtype.names
            
            # 尝试从属性读取分辨率，读不到就给一个默认值
            self.width = self.h5_file.attrs.get('width', 1280)
            self.height = self.h5_file.attrs.get('height', 800)

            palette_map = {
                "Dark": ColorPalette.Dark, "Light": ColorPalette.Light,
                "CoolWarm": ColorPalette.CoolWarm, "Gray": ColorPalette.Gray
            }
            palette = palette_map.get(palette_type, ColorPalette.Dark)

            self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=self.width, sensor_height=self.height, fps=self.fps, palette=palette)
            self.event_frame_gen.set_output_callback(self._on_cd_frame_cb)

        else:
            # ---------------- 【Metavision 引擎初始化】 ----------------
            try:
                self.device = initiate_device("")
                self.mv_iterator = EventsIterator.from_device(device=self.device, delta_t=delta_t_us)
            except:
                self.device = None
                self.mv_iterator = EventsIterator(input_path=self.input_path, delta_t=delta_t_us)
            
            self.height, self.width = self.mv_iterator.get_size()
            
            palette_map = {
                "Dark": ColorPalette.Dark, "Light": ColorPalette.Light,
                "CoolWarm": ColorPalette.CoolWarm, "Gray": ColorPalette.Gray
            }
            palette = palette_map.get(palette_type, ColorPalette.Dark)

            self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=self.width, sensor_height=self.height, fps=self.fps, palette=palette)
            self.event_frame_gen.set_output_callback(self._on_cd_frame_cb)

        # ==========================================
        # 2. 发送公共配置包
        # ==========================================
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
        """Metavision 算法生成图像后的回调"""
        if self.is_running:
            self.image_signal.emit(cd_frame.copy())

    def start_recording(self):
        if self.device is not None:  
            i_events_stream = self.device.get_i_events_stream()
            if i_events_stream:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.raw"
                i_events_stream.log_raw_data(filename)
                self.is_recording = True
                print(f"开始录制原始数据: {filename}")

    def stop_recording(self):
        """停止 RAW 录制信号"""
        if self.device is not None:
            i_events_stream = self.device.get_i_events_stream()
            if i_events_stream:
                i_events_stream.stop_log_raw_data()
                self.is_recording = False
                print("停止录制并保存文件")

    def run(self):
        # 根据当前模式，路由到对应的运行循环
        if self.is_aedat4:
            self._run_aedat4_loop()
        elif self.is_h5:
            self._run_h5_loop()
        else:
            self._run_metavision_loop()

    def _run_h5_loop(self):
        """严格按照 FPS (时间切片) 驱动的 H5 读取循环"""
        total_events = len(self.events_dataset)
        current_idx = 0

        time_key = 't' if 't' in self.h5_dtypes else ('ts' if 'ts' in self.h5_dtypes else 'timestamp')
        pol_key = 'p' if 'p' in self.h5_dtypes else ('pol' if 'pol' in self.h5_dtypes else 'polarity')

        # 计算一帧严格对应的时间跨度（微秒）
        frame_interval_us = int(1_000_000 / self.fps)
        
        start_real_time = time.perf_counter()
        start_sensor_time = None
        next_frame_target_time = None

        while self.is_running and current_idx < total_events:
            events_for_this_frame = []
            
            while current_idx < total_events:
                step = 5000
                end_idx = min(current_idx + step, total_events)
                raw_events = self.events_dataset[current_idx:end_idx]
                
                evs = np.zeros(len(raw_events), dtype=[('x', '<u2'), ('y', '<u2'), ('p', 'i1'), ('t', '<i8')])
                evs['x'] = raw_events['x']
                evs['y'] = raw_events['y']
                evs['p'] = raw_events[pol_key]
                evs['t'] = raw_events[time_key]

                if start_sensor_time is None:
                    start_sensor_time = evs['t'][0]
                    next_frame_target_time = start_sensor_time + frame_interval_us
                    start_real_time = time.perf_counter()

                over_time_indices = np.where(evs['t'] >= next_frame_target_time)[0]
                
                if len(over_time_indices) > 0:
                    split_idx = over_time_indices[0]
                    events_for_this_frame.append(evs[:split_idx])
                    current_idx = current_idx + split_idx
                    break  
                else:
                    events_for_this_frame.append(evs)
                    current_idx = end_idx

            if not events_for_this_frame:
                break
                
            frame_events = np.concatenate(events_for_this_frame)

            # 送去渲染
            if len(frame_events) > 0:
                self.event_frame_gen.process_events(frame_events)

                # 实时处理：送入神经网络
                if self.analysis_enabled and self.target_queue is not None:
                    try:
                        clean_array = np.column_stack((
                            frame_events['x'], frame_events['y'], 
                            frame_events['t'], frame_events['p']
                        )).astype(np.float32)
                        if self.target_queue.full():
                            self.target_queue.get_nowait()
                        self.target_queue.put_nowait(clean_array)
                    except queue.Full:
                        pass

            # 严格的播放速度控制
            next_frame_target_time += frame_interval_us
            sensor_elapsed_s = (next_frame_target_time - start_sensor_time) / 1_000_000.0
            real_elapsed_s = time.perf_counter() - start_real_time
            sleep_time = sensor_elapsed_s - real_elapsed_s
            
            if sleep_time > 0.005:
                time.sleep(sleep_time)
            elif sleep_time < -0.2:
                start_real_time = time.perf_counter()
                start_sensor_time = next_frame_target_time - frame_interval_us

        if hasattr(self, 'h5_file'):
            self.h5_file.close()

        if self.is_running:
            self.finished_signal.emit()

    def _run_aedat4_loop(self):
        """AEDAT4 专属数据读取循环"""
        start_real_time = time.perf_counter()  
        start_sensor_time = None
        
        frame_interval_us = int(1_000_000 / self.fps)
        next_frame_time = None
        frame_buffer = dv.EventStore()

        while self.is_running and self.dv_reader.isRunning():
            events = self.dv_reader.getNextEventBatch()
            
            if events is None:
                print("aedat4 视频播放已结束。")
                break  
                
            if events.isEmpty():
                continue  

            arr = events.numpy()

            if start_sensor_time is None:
                start_sensor_time = arr['timestamp'][0]
                next_frame_time = start_sensor_time + frame_interval_us

            # 实时处理：送入神经网络
            if self.analysis_enabled and self.target_queue is not None:
                try:
                    clean_array = np.column_stack((
                        arr['x'], arr['y'], arr['timestamp'], arr['polarity']
                    )).astype(np.float32)
                    
                    if self.target_queue.full():
                        self.target_queue.get_nowait()
                    self.target_queue.put_nowait(clean_array)
                except queue.Full:
                    pass

            # UI显示：累积事件生成画面
            frame_buffer.add(events)
            if arr['timestamp'][-1] >= next_frame_time:
                image_bgr = self.dv_visualizer.generateImage(frame_buffer)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
                if self.is_running:
                    self.image_signal.emit(image_rgb.copy())
                    
                frame_buffer = dv.EventStore()
                next_frame_time = arr['timestamp'][-1] + frame_interval_us

            # 速度控制
            current_sensor_time = arr['timestamp'][-1]
            sensor_elapsed_s = (current_sensor_time - start_sensor_time) / 1_000_000.0
            real_elapsed_s = time.perf_counter() - start_real_time
            time_diff = sensor_elapsed_s - real_elapsed_s
            
            if time_diff > 0:
                time.sleep(time_diff)
            elif time_diff < -0.2:
                start_real_time = time.perf_counter()
                start_sensor_time = current_sensor_time

        if self.is_running:
            self.finished_signal.emit()

    def _run_metavision_loop(self):
        """Metavision 原有的数据读取循环"""
        start_real_time = time.time()
        start_sensor_time = None

        for evs in self.mv_iterator:
            if not self.is_running:
                break

            if evs.size > 0 and start_sensor_time is None:
                start_sensor_time = evs['t'][0]

            # 1. UI显示
            self.event_frame_gen.process_events(evs)

            # 2. 实时处理
            if self.analysis_enabled and self.target_queue is not None:
                if evs.size > 0:
                    try:
                        clean_array = np.column_stack((
                            evs['x'], evs['y'], evs['t'], evs['p']
                        )).astype(np.float32)
                        if self.target_queue.full():
                            self.target_queue.get_nowait()
                        self.target_queue.put_nowait(clean_array)
                    except queue.Full:
                        pass
            
            # 速度控制
            if self.device is None and evs.size > 0 and start_sensor_time is not None:
                current_sensor_time = evs['t'][-1]
                sensor_elapsed_s = (current_sensor_time - start_sensor_time) / 1000000.0
                real_elapsed_s = time.time() - start_real_time
                time_diff = sensor_elapsed_s - real_elapsed_s
                if time_diff > 0:
                    time.sleep(time_diff)

        if self.is_running:
            self.finished_signal.emit()

    def stop(self):
        self.is_running = False
        if self.is_recording:
            self.is_recording = False
            self.stop_recording()