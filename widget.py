import sys
import cv2
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic

# 导入刚才写的后端类
from backend.metavision_hal_get_started import CameraThread

class MainWindow1(QWidget):
    def __init__(self):
        super().__init__()
        # 加载 UI 文件
        uic.loadUi("form.ui", self)

        self.thread = None # 相机线程句柄

        # 初始状态设置
        self.btn_record.setEnabled(False)
        self.btn_record.setText("开始录制 RAW")

        # --- 信号与槽连接 ---
        self.btn_toggle_camera.clicked.connect(self.toggle_camera)
        self.btn_record.clicked.connect(self.toggle_recording)

        # 👇 连接下拉框信号：当选择改变时触发
        self.combo_palette.currentTextChanged.connect(self.on_palette_changed)

    def on_palette_changed(self, text):
        """下拉框变化时的槽函数"""
        # 只有当线程正在运行时，才去实时修改
        # 如果线程没运行，下次启动时会自动读取 combo 的当前值
        if self.thread is not None and self.thread.isRunning():
            print(f"前端请求切换颜色: {text}")
            self.thread.set_palette(text)

    def toggle_camera(self):
        # 🟢 启动相机
        if self.thread is None:
            # 获取当前下拉框选中的颜色
            current_palette = self.combo_palette.currentText()

            # 实例化线程
            self.thread = CameraThread(current_palette)
            self.thread.image_signal.connect(self.update_image)
            self.thread.start()

            # 更新 UI
            self.btn_toggle_camera.setText("停止相机")
            self.btn_record.setEnabled(True)

        # 🔴 停止相机
        else:
            self.thread.stop()
            self.thread.deleteLater() # 清理内存
            self.thread = None

            # 更新 UI
            self.btn_toggle_camera.setText("启动相机")
            self.btn_record.setEnabled(False)
            self.btn_record.setText("开始录制 RAW")
            self.image_label.setText("相机已停止")

    def toggle_recording(self):
        if self.thread and self.thread.isRunning():
            if not self.thread.is_recording:
                self.thread.start_recording()
                self.btn_record.setText("停止录制 (正在写入...)")
                self.btn_record.setStyleSheet("color: red;") # 变红提示
            else:
                self.thread.stop_recording()
                self.btn_record.setText("开始录制 RAW")
                self.btn_record.setStyleSheet("")

    def update_image(self, cv_img):
        """接收后端发来的 numpy 数组并显示"""
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width

        # 转换为 Qt 图片格式 (Format_BGR888 对应 OpenCV 的默认顺序)
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)

        # 缩放并显示
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def closeEvent(self, event):
        """窗口关闭时确保停止线程"""
        if self.thread:
            self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow1()
    win.show()
    sys.exit(app.exec())
