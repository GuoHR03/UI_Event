import sys
import queue
import traceback
import subprocess
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog
from PyQt6 import uic
from backend.metavision_hal_get_started import CameraThread
# from backend.realtime_inference import PredictThread
from NetworkThread import NetworkThread



def exception_hook(exctype, value, tb):
    print("\n========== [!] 捕捉到致命崩溃 [!] ==========")
    traceback.print_exception(exctype, value, tb)
    print("==========================================\n")
    # 强行暂停终端，防止一闪而过
    input("程序已崩溃，请查看上方报错信息，然后按回车键退出...")
    sys.exit(1)

sys.excepthook = exception_hook


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('form.ui', self)
        self.start_btn.clicked.connect(self.toggle_camera)

        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)

        self.combo_palette.currentTextChanged.connect(self.change_camera)

        self.fpsEdit.valueChanged.connect(self.change_camera)

        self.select_file_btn.clicked.connect(self.choose_file)


        # 变量
        self.camera_thread = None
        self.file_path = None

        self.camera_queue = queue.Queue(maxsize=10) # 建议加个限制，防止生产过快
        self.network_thread = NetworkThread(self.camera_queue)
        self.network_thread.result_signal.connect(self.update_prediction_ui)
        self.network_thread.start()

        self.backend_process = None  # 用来存放 Linux 后端进程的句柄




    def toggle_camera(self):
        if self.camera_thread is None:
            # 打开wsl
            if self.backend_process is None:
                            # ！！！请务必确认这两个路径在你的 WSL 环境里是正确的！！！
                            linux_python = "/home/tianmu/anaconda3/envs/eventmamba/bin/python"
                            linux_script = "/home/tianmu/UI_Event/linux_backend.py"

                            cmd = ["wsl", linux_python, linux_script]

                            # 悄悄启动，不弹黑框
                            self.backend_process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            print("已跨界唤醒 Linux 端 EventMamba 后端")

            if self.file_path == None:
                self.camera_thread = CameraThread(
                    self.combo_palette.currentText(),
                    self.fpsEdit.value(),
                    target_queue=self.camera_queue)
            else:
                self.camera_thread = CameraThread(
                    self.combo_palette.currentText(),
                    self.fpsEdit.value(),
                    target_queue=self.camera_queue,
                    file_path = self.file_path)
            self.camera_thread.image_signal.connect(self.update_image)
            self.camera_thread.finished_signal.connect(self.on_playback_finished)
            self.camera_thread.start()
            self.start_btn.setText("停止相机")
            self.record_btn.setEnabled(True)
        else:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread.deleteLater()
            self._dying_thread = self.camera_thread
            self.camera_thread = None
            self.start_btn.setText("启动相机")
            self.record_btn.setEnabled(False)
            self.record_btn.setText("开始录制")
            self.image_label.setText("相机已停止")


            #关闭wsl
            if self.backend_process:
                            self.backend_process.terminate()
                            self.backend_process.wait()
                            self.backend_process = None
                            print("Linux 端后端已关闭，显存已释放")


    def toggle_recording(self):
        if self.camera_thread and self.camera_thread.isRunning():
            if not self.camera_thread.is_recording:
                self.camera_thread.start_recording()
                self.record_btn.setText("停止录制 (正在写入...)")
                self.record_btn.setStyleSheet("background-color: red; color: white;")
            else:
                self.camera_thread.stop_recording()
                self.record_btn.setText("开始录制 RAW")
                self.record_btn.setStyleSheet("")

    def update_image(self, cv_img):
        if len(cv_img.shape) == 3:
            """RGB"""
            height, width, channel = cv_img.shape
            bytes_per_line = channel * width
            img_format = QImage.Format.Format_BGR888
        else:
            """Gray"""
            height, width = cv_img.shape
            bytes_per_line = width
            img_format = QImage.Format.Format_Grayscale8

        q_img = QImage(cv_img.data, width, height, bytes_per_line, img_format)

        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if hasattr(self, 'network_thread'):
                    self.network_thread.stop()
                    self.network_thread.wait()
        event.accept()

    def change_camera(self):
            if self.camera_thread and self.camera_thread.isRunning():
                self.toggle_camera() # 停止旧相机

                QApplication.processEvents()

                self.toggle_camera() # 带着新选的颜色重新启动相机

    def update_prediction_ui(self,result):
        self.textEdit.setText(result)

    def on_playback_finished(self):
            """文件播放完后的自动处理"""
            print("文件播放结束")
            self.stop_camera_logic()

    def stop_camera_logic(self):
            """抽离出来的停止逻辑"""
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread.deleteLater()
                self._dying_thread = self.camera_thread
                self.camera_thread = None
            self.start_btn.setText("启动相机")
            self.record_btn.setEnabled(False)
            self.image_label.setText("播放已结束/相机已停止")
            if self.backend_process:
                    print("[System] 正在强制回收 Linux 后端...")
                    try:
                        # 直接暴力杀掉，不再温柔地 terminate
                        self.backend_process.kill()
                        # 只等 0.5 秒，不等了直接走
                        self.backend_process.wait(timeout=0.5)
                    except Exception:
                        pass
                    self.backend_process = None
                    print("Linux 后端已回收")


    def choose_file(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择离线视频文件",
                "",
                "RAW 视频 (*.raw);;所有文件 (*)"
            )

            if file_path:
                self.file_path = file_path
                self.file_path_label.setText(file_path)

                # 修改
                if self.camera_thread and self.camera_thread.isRunning():
                    self.toggle_camera()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
