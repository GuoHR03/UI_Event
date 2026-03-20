import zmq
import queue
from PyQt6.QtCore import QThread, pyqtSignal

class NetworkThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, input_queue):
        super().__init__()
        self.input_queue = input_queue
        self.running = True

        # 建立网络客户端，连接到 WSL2
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ) # REQ: 请求模式 (Request)
        # WSL2 默认会将内部端口映射到 Windows 的 localhost
        self.socket.connect("tcp://127.0.0.1:5555")


    def run(self):
        while self.running:
            try:
                # 1. 从相机的 Queue 里拿到最新的帧/事件数据 (设置超时防止阻塞退出)
                data = self.input_queue.get(timeout=1.0)

                # 2. 把数据序列化并发送给 Linux
                self.socket.send_pyobj(data)

                # 3. 等待 Linux 的 EventMamba 返回结果
                result = self.socket.recv_string()

                # 4. 把结果发送给 UI 更新
                self.result_signal.emit(result)

            except queue.Empty:
                continue # 队列没数据，继续循环
            except Exception as e:
                self.result_signal.emit(f"网络通信错误: {str(e)}")

    def stop(self):
        self.running = False
        self.socket.close(linger=0)
        self.context.term()
