# import zmq
# import subprocess
# import sys
# from backend.realtime_inference import EventMambaPredictor
# from typing import Optional

# def get_pid_by_port(port: int) -> Optional[int]: 
#     """Linux下根据端口号查找占用进程的 PID"""
#     try:
#         result = subprocess.check_output(
#             ["lsof", "-ti", f":{port}"],
#             stderr=subprocess.STDOUT,
#             text=True
#         )
#         return int(result.strip())
#     except subprocess.CalledProcessError:
#         return None
#     except Exception:
#         return None


# def kill_process(pid: int):
#     """Linux下强制终止指定 PID 的进程"""
#     try:
#         subprocess.run(["kill", "-9", str(pid)], check=True)
#     except Exception as e:
#         sys.exit(1)



# def main():

#     port = 5555
#     pid = get_pid_by_port(port)
#     if pid:
#         kill_process(pid)
#     model = EventMambaPredictor() 
#     context = zmq.Context()
#     socket = context.socket(zmq.REP)
#     socket.setsockopt(zmq.LINGER, 0)
#     socket.bind("tcp://0.0.0.0:5555")

#     while True:
#         try:
#             data = socket.recv_pyobj()   # data是numpy数组
#             result_text = model.process_data(data)
#             socket.send_string(result_text)
            
#         except Exception as e:
#             socket.send_string(f"Linux 推理报错: {str(e)}")

# if __name__ == "__main__":
#     main()


import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import torch
from backend.realtime_inference import EventMambaPredictor

class InferenceServer:
    def __init__(self, weight_path, port=5555):
        self.port = port
        self.weight_path = weight_path
        self.config = {}

        print(f" [初始化] 正在加载模型权重: {self.weight_path} ...")
        try:
            self.model = EventMambaPredictor()
            print("[初始化] 模型加载完成，显存已分配。")
        except Exception as e:
            print(f"[致命错误] 模型加载失败: {e}")
            sys.exit(1)

        handler = self._build_handler()
        self.httpd = ThreadingHTTPServer(("", self.port), handler)
        print(f" [就绪] HTTP 服务端已绑定端口 {self.port}，等待请求...")

    def _build_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._send_json(200, {"status": "ok"})
                else:
                    self._send_json(404, {"error": "not_found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b""
                try:
                    payload = json.loads(body.decode("utf-8")) if body else {}
                except Exception:
                    self._send_json(400, {"error": "bad_json"})
                    return

                if self.path == "/config":
                    try:
                        result = server.handle_config(payload)
                        self._send_json(200, result)
                    except Exception as e:
                        self._send_json(500, {"error": str(e)})
                    return

                if self.path == "/infer":
                    try:
                        result_text = server.handle_infer(payload)
                        self._send_json(200, {"result": result_text})
                    except Exception as e:
                        self._send_json(500, {"error": str(e)})
                    return

                self._send_json(404, {"error": "not_found"})

            def _send_json(self, status_code, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                return

        return Handler

    def handle_config(self, config_data):
        self.config = config_data
        return {"status": "ok"}

    def handle_infer(self, payload):
        data_b64 = payload.get("data_b64")
        shape = payload.get("shape")
        dtype = payload.get("dtype")
        if not data_b64 or not shape or not dtype:
            raise ValueError("invalid_payload")
        raw = base64.b64decode(data_b64)
        array = np.frombuffer(raw, dtype=np.dtype(dtype))
        array = array.reshape(shape)
        return self.model.process_data(array)

    def run(self):
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(" [退出] 引擎已安全关闭，资源已释放。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EventMamba Linux Backend Server")
    parser.add_argument("--weights", type=str, required=True, help="模型 .pt 权重文件的路径")
    parser.add_argument("--port", type=int, default=5555, help="HTTP 绑定的端口号")
    args = parser.parse_args()

    server = InferenceServer(weight_path=args.weights, port=args.port)
    
    try:
        server.run()
    except KeyboardInterrupt:
        server.stop()
