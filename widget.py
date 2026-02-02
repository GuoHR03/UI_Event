import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

# 确保 backend 文件夹里有 camera.py
from backend.metavision_hal_get_started import CameraThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 600)
        self.setWindowTitle("Prophesee Minimal Viewer")

        # 布局设置
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)

        # 1. 画面显示区域
        self.image_label = QLabel("等待相机启动...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: black; color: white; font-size: 20px;")
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label)

        # 2. 唯一的按钮
        self.btn_startCamera = QPushButton("启动相机")
        self.btn_startCamera.clicked.connect(self.toggle_camera)
        self.btn_startCamera.setFixedHeight(50) # 按钮大一点
        layout.addWidget(self.btn_startCamera)

        # 3. 唯一的按钮
        self.combo = QComboBox()
        self.combo.addItem("Dark")
        self.combo.addItem("Light")
        self.combo.addItem("CoolWarm")
        self.combo.addItem("Gray")
        self.combo.setFixedHeight(50) # 按钮大一点
        layout.addWidget(self.combo)

        # 4. 唯一的按钮
        self.btn_startRecord = QPushButton("开始录制")
        self.btn_startRecord.clicked.connect(self.record)
        self.btn_startRecord.setFixedHeight(50) # 按钮大一点
        layout.addWidget(self.btn_startRecord)
        self.thread = None

    def toggle_camera(self):
        if self.thread == None:
            self.thread = CameraThread(self.combo.currentText())
            self.thread.set_palette(self.combo.currentText())
            self.thread.image_signal.connect(self.update_image)
            self.thread.start()
            self.btn_startCamera.setText("停止相机")
        else:
            # === 停止 ===
            self.thread.stop()
            self.thread = None
            self.btn_startCamera.setText("启动相机")
            self.image_label.setText("相机已停止")


    def record(self):
        print("开始录制")

    def update_image(self, cv_img):
        """显示图像的核心函数"""
        # OpenCV 的 BGR 转为 Qt 的 RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # 缩放并显示
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        """退出时强制清理，防止下次无法打开"""
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
