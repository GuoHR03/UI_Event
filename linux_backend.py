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


import zmq
import argparse
import sys
import torch
from backend.realtime_inference import EventMambaPredictor

class InferenceServer:
    def __init__(self, weight_path, port=5555):
        self.port = port
        self.weight_path = weight_path
        self.running = True
        
        print(f" [初始化] 正在加载模型权重: {self.weight_path} ...")  #删除
        # 1. 初始化模型（放到类的属性里）
        try:
            # 假设你的模型支持传入权重路径
            #self.model = EventMambaPredictor(weight_path=self.weight_path) 
            self.model = EventMambaPredictor() 
            print("[初始化] 模型加载完成，显存已分配。") #删除
        except Exception as e:
            print(f"[致命错误] 模型加载失败: {e}")
            sys.exit(1)

        # 2. 初始化网络通信
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 1)    # 防止积压
        self.socket.setsockopt(zmq.CONFLATE, 1)  # 只拿最新帧
        self.socket.bind(f"tcp://0.0.0.0:{self.port}")
        print(f" [就绪] ZMQ 服务端已绑定端口 {self.port}，等待请求...") #删除

    def run(self):
        """核心运行循环"""
        while self.running:
            try:
                # 1. 接收请求
                data = self.socket.recv_pyobj()

                # 2. 区分是控制指令还是推理数据
                if isinstance(data, dict) and data.get("msg_type") == "CONFIG":
                    self.handle_config(data)
                    continue

                # 3. 核心推理
                result_text = self.model.process_data(data)
                
                # 4. 返回结果
                self.socket.send_string(result_text)

            except Exception as e:
                print(f" [警告] 推理循环出错: {e}")  #删除
                # 保证 REQ/REP 模式下哪怕报错也必须回传，防止 Windows 卡死
                try:
                    self.socket.send_string(f"Error: {str(e)}")
                except:
                    pass

    def handle_config(self, config_data):
        """专门处理 Windows 发来的配置信息"""
        print(f"[配置更新] 收到新参数: {config_data}")  #删除
        # 比如：self.model.update_resolution(config_data['width'], config_data['height'])
        self.socket.send_string("CONFIG_OK")

    def stop(self):
        """安全释放资源"""
        self.running = False
        self.socket.close()
        self.context.term()
        # 清理 PyTorch 显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(" [退出] 引擎已安全关闭，资源已释放。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EventMamba Linux Backend Server")
    parser.add_argument("--weights", type=str, required=True, help="模型 .pt 权重文件的路径")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ 绑定的端口号")
    args = parser.parse_args()

    server = InferenceServer(weight_path=args.weights, port=args.port)
    
    try:
        server.run()
    except KeyboardInterrupt:
        # 捕捉 Ctrl+C 强制退出（或者 Windows 端发来的 kill 信号）
        server.stop()