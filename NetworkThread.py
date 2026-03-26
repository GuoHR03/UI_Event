#Windows端
import base64
import json
import queue
import urllib.error
import urllib.request
from PyQt6.QtCore import QThread, pyqtSignal

class NetworkThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, input_queue, base_url="http://127.0.0.1:5555"):
        super().__init__()
        self.input_queue = input_queue
        self.running = True
        self.base_url = base_url.rstrip("/")
        self.infer_url = f"{self.base_url}/infer"
        self.config_url = f"{self.base_url}/config"

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

                if isinstance(data, dict) and data.get("msg_type") == "CONFIG":
                    self._post_json(self.config_url, data)
                    continue

                result = self._post_infer(data)
                if result is not None:
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

    def _post_infer(self, data):
        payload = self._encode_array_payload(data)
        response = self._post_json(self.infer_url, payload)
        if not response:
            return None
        return response.get("result")

    def _encode_array_payload(self, data):
        if data is None:
            return {}
        raw = data.tobytes()
        return {
            "data_b64": base64.b64encode(raw).decode("utf-8"),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
        }

    def _post_json(self, url, payload):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                raw = response.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
        except urllib.error.URLError:
            return {}
