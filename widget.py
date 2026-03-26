import sys
import traceback
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog
from PyQt6 import uic
from backend.api import BackendAPI


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
        self.backend = BackendAPI()
        self.backend.image_signal.connect(self.update_image)
        self.backend.prediction_signal.connect(self.update_prediction_ui)
        self.backend.playback_finished_signal.connect(self.on_playback_finished)
        self.file_path = None
        self.pt_path = None

    def toggle_camera(self):
        """相机启动链接的槽"""
        if not self.backend.is_camera_running():
            self.backend.start_camera(self.combo_palette.currentText(), self.fpsEdit.value())
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
        if self.backend.is_camera_running() and self.backend.camera_thread:
            if not self.backend.camera_thread.is_recording:
                self.backend.start_recording()
                self.record_btn.setText("停止录制 (正在写入...)")
                self.record_btn.setStyleSheet("background-color: red; color: white;")
            else:
                self.backend.stop_recording()
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
        self.backend.close()
        event.accept()

    def change_camera(self):
        """更改相机参数"""
        if self.backend.is_camera_running():
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

    def stop_camera_logic(self):
        """抽离出来的停止逻辑"""
        self.backend.stop_camera()

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
            self.backend.set_input_file(file_path)
            self.file_path_label.setText(file_path)
            # 修改
            if self.backend.is_camera_running():
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
                if self.backend.is_camera_running():
                    self.toggle_camera()


    def load_Eventmamba(self):
        """加载Eventmamba模型以及网络通信"""
        print("加载")
        if self.pt_path == None:
            print("没有加载权重")
        else:
            self.backend.start_eventmamba(self.pt_path)
        self.apply_pt_btn.setEnabled(False)
        self.close_pt_btn.setEnabled(True)

    def unload_Eventmamba(self):
        """关闭Eventmamba模型 以及WSL"""
        print("关闭权重WSL")
        self.backend.stop_eventmamba()
        self.apply_pt_btn.setEnabled(True)
        self.close_pt_btn.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
