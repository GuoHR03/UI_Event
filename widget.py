import sys
import queue
import traceback
import subprocess
import os
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog
from PyQt6 import uic
from backend.Camera import CameraThread
# from backend.realtime_inference import PredictThread
from NetworkThread import NetworkThread


def exception_hook(exctype, value, tb):
    print("\n========== [!] 捕捉到致命崩溃 [!] ==========")
    traceback.print_exception(exctype, value, tb)
    print("==========================================\n")
    input("程序已崩溃，请查看上方报错信息，然后按回车键退出...")
    sys.exit(1)

sys.excepthook = exception_hook


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('form.ui', self)

        #信号与槽
        self.start_btn.clicked.connect(self.toggle_camera)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        self.combo_palette.currentTextChanged.connect(self.change_camera)
        self.fpsEdit.valueChanged.connect(self.change_camera)
        self.select_pt_btn.clicked.connect(self.choose_file_pt)
        self.apply_pt_btn.clicked.connect(self.load_Eventmamba)
        self.close_pt_btn.clicked.connect(self.unload_Eventmamba)
        self.select_file_btn.clicked.connect(self.choose_file)

        # 变量
        self.camera_thread = None
        self.file_path = None
        self.pt_path = None
        self.camera_queue = queue.Queue(maxsize=1)
        self.network_thread = None
        #self.network_thread = NetworkThread(self.camera_queue)
        #self.network_thread.result_signal.connect(self.update_prediction_ui)
        #self.network_thread.start()
        self.backend_process = None  # 用来存放 Linux 后端进程的句柄

    def toggle_camera(self):
        """相机启动链接的槽"""
        if self.camera_thread is None:
            """启动相机或者导入文件"""
            if self.file_path == None:
                """没有RAW文件，启动相机"""
                self.camera_thread = CameraThread(
                    self.combo_palette.currentText(),
                    self.fpsEdit.value(),
                    target_queue=self.camera_queue)

            else:
                """载入RAW文件"""
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
            """关闭相机"""
            self.stop_camera_logic()
            """
            self.camera_thread.stop() #发出停止指令
            self.camera_thread.wait() #彻底执行完run()
            self.camera_thread.deleteLater() #发出删除对象指令
            self._dying_thread = self.camera_thread #把旧进程放在此处，等到下次的时候，才会完全被清理
            self.camera_thread = None
            self.start_btn.setText("启动相机")
            self.record_btn.setEnabled(False)
            self.record_btn.setText("开始录制")
            self.image_label.setText("相机已停止")
            """

    def toggle_recording(self):
        if self.camera_thread and self.camera_thread.isRunning():
            if not self.camera_thread.is_recording:
                self.camera_thread.start_recording()
                self.record_btn.setText("停止录制 (正在写入...)")
                self.record_btn.setStyleSheet("background-color: red; color: white;")
            else:
                self.camera_thread.stop_recording()
                self.record_btn.setText("开始录制")
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
        """关闭程序"""
        if self.camera_thread:
            self.camera_thread.stop()
        if self.network_thread is not None:
            self.network_thread.stop()
            self.network_thread.wait()
        if self.backend_process:
            self.backend_process.kill()
            subprocess.run(["wsl", "pkill", "-9", "-f", "linux_backend.py"])
        event.accept()

    def change_camera(self):
        """更改相机参数"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.toggle_camera() # 停止旧相机
            QApplication.processEvents()
            self.toggle_camera() # 带着新选的颜色重新启动相机

    def update_prediction_ui(self,result):
        """更新网络输出到UI文本框"""
        self.textEdit.append(result)
        #self.textEdit.setText(result)



    #  从这里继续查代码
    def on_playback_finished(self):
        """文件播放完后的自动处理"""
        self.stop_camera_logic()

    # def stop_camera_logic(self):
    #     #抽离出来的停止逻辑#
    #     if self.camera_thread:
    #         self.camera_thread.stop()
    #         self.camera_thread.deleteLater()
    #         #self._dying_thread = self.camera_thread
    #         #self.camera_thread = None
    #     self.start_btn.setText("启动相机")
    #     self.record_btn.setEnabled(False)
    #     self.image_label.setText("播放已结束/相机已停止")
    #     if self.backend_process:
    #             try:
    #                 # 直接暴力杀掉，不再温柔地 terminate
    #                 self.backend_process.kill()
    #                 # 只等 0.5 秒，不等了直接走
    #                 self.backend_process.wait(timeout=0.5)
    #             except Exception:
    #                 pass
    #             self.backend_process = None

    def stop_camera_logic(self):
        """抽离出来的停止逻辑"""
        if self.camera_thread:
            self.camera_thread.stop()
            if not self.camera_thread.wait(1000):
                print("警告：相机底层卡死，正在强制终止线程！")
                self.camera_thread.terminate()
                self.camera_thread.wait(500) # 强杀后再等0.5秒让系统回收资源

            self.camera_thread.deleteLater()

            self.camera_thread = None

        self.start_btn.setText("启动相机")
        self.record_btn.setEnabled(False)
        self.image_label.setText("播放已结束/相机已停止")

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

    def choose_file_pt(self):
            pt_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择权重文件",
                "",
                "pt (*.pt);;所有文件 (*)"
            )
            if pt_path:
                self.pt_path = pt_path
                self.pt_path_label.setText(pt_path)
                # 修改
                if self.camera_thread and self.camera_thread.isRunning():
                    self.toggle_camera()


    def load_Eventmamba(self):
        """加载Eventmamba模型以及网络通信"""
        print("加载")
        if self.pt_path == None:
            print("没有加载权重")
        else:
            """需要启动WSL加载模型"""
            if self.backend_process is None or self.backend_process.poll() is not None:
                """没有后端进程或者后端进程已死,这时候会重新创建后端进程"""
                current_dir = os.path.dirname(os.path.abspath(__file__))
                linux_python = "/home/tianmu/anaconda3/envs/eventmamba/bin/python"
                linux_script = "linux_backend.py"
                cmd = ["wsl", linux_python, linux_script,"--weights",self.pt_path]
                self.backend_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=current_dir
                )
        self.apply_pt_btn.setEnabled(False)
        self.close_pt_btn.setEnabled(True)
        if self.network_thread is None:
            self.network_thread = NetworkThread(self.camera_queue)
            self.network_thread.result_signal.connect(self.update_prediction_ui)
            self.network_thread.start()

    def unload_Eventmamba(self):
        """关闭Eventmamba模型 以及WSL"""
        print("关闭权重WSL")

        if self.network_thread:
            self.network_thread.stop()
            self.network_thread.wait()
            self.network_thread.deleteLater()
            self.network_thread = None
        #关闭wsl   需要修改
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait()
            self.backend_process = None
            try:
                subprocess.run(
                    ["wsl", "pkill", "-f", "linux_backend.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"清理 Linux 后端时出现小问题: {e}")
        self.apply_pt_btn.setEnabled(True)
        self.close_pt_btn.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())