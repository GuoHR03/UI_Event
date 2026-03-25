#Windows端
import time
import zmq
import queue
from PyQt6.QtCore import QThread, pyqtSignal

class NetworkThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, input_queue):
        super().__init__()
        self.input_queue = input_queue
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:5555")

    def run(self):
        """与linux通信"""
        while self.running:
            try:

                data = None
                try:
                    while True:
                        data = self.input_queue.get_nowait()
                except queue.Empty:
                    pass

                if data is None:
                    # 如果刚才队列本来就是空的，那就老老实实阻塞等下一帧
                    data = self.input_queue.get(timeout=1.0)

                # 2. 发送给 Linux 推理
                self.socket.send_pyobj(data)
                result = self.socket.recv_string()
                self.result_signal.emit(result)
            #     data = self.input_queue.get(timeout=1.0)

            #     self.socket.send_pyobj(data)
            #     result = self.socket.recv_string()
            #     self.result_signal.emit(result)

            except queue.Empty:
                continue
            except Exception as e:
                    print(f"通信链路故障...")

    def stop(self):
        self.running = False
        self.socket.close(linger=0)
        self.context.term()
