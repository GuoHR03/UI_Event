import os
import sys
import time
import queue
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

# 确保能找到 models 文件夹
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(BASE_DIR, 'models') not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, 'models'))

from models.eventmamba_v1 import EventMamba

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

class PredictThread(QThread):
    # 预测结果信号：返回 格式化字符串, X坐标, Y坐标
    result_signal = pyqtSignal(str, float, float) 

    def __init__(self, model_path='./checkpoint/ini30/v2/P3best_checkpoint.pth', num_point=1024, device='cuda'):
        super().__init__()
        # 使用 maxsize=1 确保队列里只有最新的一包数据，避免积压导致延迟
        self.input_queue = queue.Queue(maxsize=1) 
        self.is_running = True
        self.num_point = num_point
        
        # 自动选择设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # ---------------- 模型初始化 ----------------
        print(f"Initializing EventMamba on {self.device}...")
        self.model = EventMamba(num_classes=2)
        self.model.apply(inplace_relu)
        
        # 加载权重
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: Checkpoint not found at {model_path}. Using random weights.")
            
        self.model.to(self.device)
        self.model.eval() # 务必设置为 eval 模式
        # ---------------------------------------------

    def preprocess(self, raw_evs):
        """
        将原始事件数据预处理为模型需要的 Tensor 格式
        假设 raw_evs 是一个包含 [x, y, p, t] 的列表或 numpy 数组
        """
        evs_array = np.array(raw_evs, dtype=np.float32)
        N = evs_array.shape[0]

        if N == 0:
            return None

        # 1. 重采样到固定的 num_point (1024)
        if N >= self.num_point:
            # 数据量足够时，随机下采样 (无放回)
            indices = np.random.choice(N, self.num_point, replace=False)
            sampled_evs = evs_array[indices]
        else:
            # 数据量不足时，有放回重采样补齐点数
            indices = np.random.choice(N, self.num_point, replace=True)
            sampled_evs = evs_array[indices]

        # 2. 维度转换
        data_tensor = torch.from_numpy(sampled_evs) # 形状: [1024, 4]
        data_tensor = data_tensor.unsqueeze(0)      # 增加 Batch 维度: [1, 1024, 4]
        data_tensor = data_tensor.permute(0, 2, 1)  # 调整为模型输入: [1, 4, 1024]
        
        return data_tensor.to(self.device)

    def run(self):
        while self.is_running:
            try:
                # 拿到的是原始事件包 (包含 x, y, p, t)
                raw_evs = self.input_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # --- 1. 预处理 ---
                input_tensor = self.preprocess(raw_evs)
                if input_tensor is None:
                    continue
                
                # --- 2. 神经网络推理 ---
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                
                # --- 3. 后处理与结果分发 ---
                # outputs 形状为 [1, 2]
                pred = outputs.cpu().numpy()[0]
                pred_x, pred_y = float(pred[0]), float(pred[1])
                
                # 计算处理延迟
                latency = (time.time() - start_time) * 1000
                
                # 组装预测结果
                res_text = f"检测到 {len(raw_evs)} 个点 | 瞳孔中心: ({pred_x:.1f}, {pred_y:.1f}) | 耗时: {latency:.1f}ms"
                
                # 发射信号 (把格式化文本和具体坐标都发出去，方便 GUI 画图或打印)
                self.result_signal.emit(res_text, pred_x, pred_y)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")

    def stop(self):
        self.is_running = False
        self.wait()