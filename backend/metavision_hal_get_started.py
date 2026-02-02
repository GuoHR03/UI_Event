import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from metavision_hal import DeviceDiscovery
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

class CameraThread(QThread):
    # 只保留图像信号，删除了日志信号
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self,palette = "Dark"):
        super().__init__()
        self.is_running = False
        self.device = None
        self.palette_dir = {"Dark":ColorPalette.Dark,
                            "Light":ColorPalette.Light,
                            "CoolWarm":ColorPalette.CoolWarm,
                            "Gray":ColorPalette.Gray
        }
        self.palette = self.palette_dir[palette]

    def run(self):
        # 提前定义变量，防止 finally 报错
        i_eventsstream = None
        
        try:
            # 1. 打开相机
            self.device = DeviceDiscovery.open("")
            
            # 2. 获取组件
            i_eventsstream = self.device.get_i_events_stream()
            i_eventsstreamdecoder = self.device.get_i_events_stream_decoder()
            i_cddecoder = self.device.get_i_event_cd_decoder()
            geometry = self.device.get_i_geometry() 
            
            # 3. 初始化算法 (固定为 Dark 风格，最基础)
            self.frame_gen = PeriodicFrameGenerationAlgorithm(
                geometry.get_width(), geometry.get_height(), 
                fps=30, palette=self.palette
            )
            
            # 绑定回调：生成的图片通过信号发出去
            self.frame_gen.set_output_callback(lambda ts, frame: self.image_signal.emit(frame))

            # 绑定解码器
            def on_cd_events(event_buffer):
                if event_buffer.size > 0:
                    self.frame_gen.process_events(event_buffer)

            i_cddecoder.add_event_buffer_callback(on_cd_events)

            # 4. 启动流
            i_eventsstream.start()
            self.is_running = True

            # 5. 循环读取
            while self.is_running:
                ret = i_eventsstream.poll_buffer()
                if ret < 0:
                    break
                elif ret > 0:
                    raw_data = i_eventsstream.get_latest_raw_data()
                    if raw_data is not None:
                        i_eventsstreamdecoder.decode(raw_data)
        
        except Exception as e:
            print(f"相机运行出错: {e}")

        finally:
            # ==========================================
            # 就算是最简版，这部分也不能删！
            # 否则你的相机只能开一次，第二次必报错。
            # ==========================================
            print("正在清理资源...")
            if i_eventsstream is not None:
                try:
                    i_eventsstream.stop()
                except:
                    pass
            
            # 按顺序销毁对象，防止闪退
            if hasattr(self, 'frame_gen'):
                del self.frame_gen
            
            if self.device is not None:
                del self.device 
                self.device = None
                print("设备已安全释放")

    def stop(self):
        self.is_running = False
        self.wait()
    
    def set_palette(self,palette):
        self.palette = self.palette_dir[palette]
        print(self,palette)
