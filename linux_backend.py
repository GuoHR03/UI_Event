# ==========================================
# 这个文件叫 linux_backend.py，只在 WSL2 里运行！
# ==========================================
import zmq
import torch
import sys

# 👉 看这里！EventMamba 是在这里被导入和加载的！
from backend.realtime_inference import PredictThread # 或者你实际的模型类

def main():
    print("🚀 WSL2 后端启动中，正在加载 EventMamba 模型...")
    
    # 1. 实例化你的大模型 (放到 GPU 里)
    # 注意：你需要把你原来的 PredictThread 稍微改一下，让它变成一个普通的推理类，而不是 QThread
    model = PredictThread() 
    
    # 2. 建立接收端口 (厨房传菜口)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5555")
    print("✅ EventMamba 已就绪！正在监听 Windows 发来的数据...")

    while True:
        try:
            # 3. 接收 Windows 发来的相机数据
            data = socket.recv_pyobj()
            
            # 4. 把数据喂给 EventMamba 进行推理！
            # 假设你的模型里有一个处理数据的方法叫 process_data
            result_text = model.process_data(data) 
            
            # 5. 把推理出来的字符串结果发回给 Windows 界面
            socket.send_string(result_text)
            
        except Exception as e:
            socket.send_string(f"Linux 推理报错: {str(e)}")

if __name__ == "__main__":
    main()