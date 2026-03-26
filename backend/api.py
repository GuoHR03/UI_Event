import os
import queue
import subprocess

from PyQt6.QtCore import QObject, pyqtSignal

from backend.Camera import CameraThread
from NetworkThread import NetworkThread


class BackendAPI(QObject):
    image_signal = pyqtSignal(object)
    prediction_signal = pyqtSignal(str)
    playback_finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.camera_queue = queue.Queue(maxsize=1)
        self.camera_thread = None
        self.network_thread = None
        self.backend_process = None
        self.file_path = None

    def is_camera_running(self):
        return self.camera_thread is not None and self.camera_thread.isRunning()

    def is_inference_running(self):
        return self.network_thread is not None and self.network_thread.isRunning()

    def set_input_file(self, file_path):
        self.file_path = file_path

    def start_camera(self, palette, fps):
        if self.camera_thread is not None:
            self.stop_camera()

        kwargs = {
            "palette_type": palette,
            "fps": fps,
            "target_queue": self.camera_queue,
        }
        if self.file_path:
            kwargs["file_path"] = self.file_path

        self.camera_thread = CameraThread(**kwargs)
        self.camera_thread.image_signal.connect(self.image_signal.emit)
        self.camera_thread.finished_signal.connect(self.playback_finished_signal.emit)
        self.camera_thread.start()

    def stop_camera(self):
        if not self.camera_thread:
            return

        self.camera_thread.stop()
        if not self.camera_thread.wait(1000):
            self.camera_thread.terminate()
            self.camera_thread.wait(500)

        self.camera_thread.deleteLater()
        self.camera_thread = None

    def start_recording(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.start_recording()

    def stop_recording(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop_recording()

    def start_eventmamba(self, weights_path, port=5555, host="127.0.0.1"):
        if not weights_path:
            raise ValueError("weights_path is required")

        if self.backend_process is None or self.backend_process.poll() is not None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(current_dir)
            linux_python = "/home/tianmu/anaconda3/envs/eventmamba/bin/python"
            linux_script = "linux_backend.py"
            cmd = ["wsl", linux_python, linux_script, "--weights", weights_path, "--port", str(port)]
            self.backend_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=project_dir,
            )

        if self.network_thread is None or not self.network_thread.isRunning():
            base_url = f"http://{host}:{port}"
            self.network_thread = NetworkThread(self.camera_queue, base_url=base_url)
            self.network_thread.result_signal.connect(self.prediction_signal.emit)
            self.network_thread.start()

    def stop_eventmamba(self):
        if self.network_thread:
            self.network_thread.stop()
            self.network_thread.wait()
            self.network_thread.deleteLater()
            self.network_thread = None

        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=2)
            except Exception:
                try:
                    self.backend_process.kill()
                except Exception:
                    pass
            self.backend_process = None

        try:
            subprocess.run(
                ["wsl", "pkill", "-f", "linux_backend.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def close(self):
        self.stop_camera()
        self.stop_eventmamba()
