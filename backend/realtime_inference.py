from PyQt6.QtCore import QThread, pyqtSignal
import queue

class PredictThread(QThread):
    # 预测结果信号，返回字符串给 UI 界面显示
    result_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # 队列用于接收原始事件包，maxsize=1 保证只处理最新的数据
        self.input_queue = queue.Queue(maxsize=1)
        self.is_running = True

    def run(self):
        while self.is_running:
            try:
                # 尝试从队列获取数据，设置超时防止死锁
                raw_evs = self.input_queue.get(timeout=0.1)

                # ---- 在这里调用你的模型推理逻辑 ----
                # 1. 预处理（例如转为 Tensor 或 Event Stack）
                # 2. 推理：output = model(tensor)
                # ----------------------------------

                # 模拟预测结果并发射信号
                res_text = f"检测到 {len(raw_evs)} 个事件点"
                self.result_signal.emit(res_text)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"推理线程出错: {e}")
                continue

    def stop(self):
        self.is_running = False
        self.wait()