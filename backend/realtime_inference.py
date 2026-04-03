import torch
import numpy as np
import os
import sys
current_root = os.getcwd() 
modules_parent_dir = os.path.join(current_root, "backend", "Eventmamba", "models")

if modules_parent_dir not in sys.path:
    sys.path.append(modules_parent_dir)
from backend.Eventmamba.models.eventmamba_v1 import EventMamba

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def downSampling_and_normalization(data_numpy,src_width=640,src_height=480, dst_width=640, dst_height=480):
    """进行下采样,适用于seet数据集"""
    x_values = data_numpy[:, 0] * (dst_width / src_width)
    y_values = data_numpy[:, 1] * (dst_height / src_height)
    t_values = data_numpy[:, 2]
    x_values = np.clip(x_values, 0, dst_width - 1)
    y_values = np.clip(y_values, 0, dst_height - 1)
    x_values = x_values / dst_width
    y_values = y_values / dst_height

    # t归一化处理
    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    return x_values,y_values,t_values

def downSampling_cropping_and_normalization(data_numpy,src_width=640,src_height=480, dst_width=512, dst_height=512):
    """进行下采样,适用于ini30数据集"""
    #缩放到 640x480 基准
    x_raw = data_numpy[:, 0] * (640.0 / src_width)
    y_raw = data_numpy[:, 1] * (480.0 / src_height)
    x_raw = np.clip(x_raw, 0, 640 - 1)
    y_raw = np.clip(y_raw, 0, 480 - 1)

    #裁剪
    mask = (x_raw >= 96) & (x_raw <= 608)
    x_values = x_raw[mask] - 96
    y_values = y_raw[mask] + 16
    t_values = data_numpy[:, 2][mask]
    
    x_values = np.clip(x_values, 0, dst_width - 1)
    y_values = np.clip(y_values, 0, dst_height - 1)

    #归一化
    x_values = x_values / dst_width
    y_values = y_values / dst_height
    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    return x_values, y_values,t_values
def process_evs_numpy2tensor(data_numpy, src_width=640, src_height=480, dst_width=640, dst_height=480, sample_size=1024):
    """处理numpy的数据,输出tensor数据,中间会对数据进行sort分类,以及归一化"""

    x_values,y_values,t_values = downSampling_and_normalization(data_numpy,src_width,src_height, dst_width, dst_height)
    #x_values,y_values,t_values = downSampling_cropping_and_normalization(data_numpy,src_width,src_height, dst_width, dst_height)
    
    # 选取1024个点
    current_sample_size = len(t_values)
    indices = np.random.choice(current_sample_size, sample_size, replace=False)
    indices = np.sort(indices)
    x = x_values[indices]
    y = y_values[indices]
    t = t_values[indices]
    data  = np.stack((t,x,y), axis=-1)
    data_tensor = torch.from_numpy(data).unsqueeze(0).permute(0, 2, 1)
    return data_tensor

class EventMambaPredictor:
    def __init__(self, weights_path):
        self.width = 640
        self.height = 480
        self.model = EventMamba(num_classes=2).cuda()
        self.load_message = ""
        try:
            state_dict = torch.load(weights_path, map_location='cuda')
            self.model.load_state_dict(state_dict)
            self.load_message = f"成功加载权重: {weights_path}"
        except Exception as e:
            self.load_message = f"权重加载失败，请检查路径或文件: {e}"
        self.model.eval()
        self.model.apply(inplace_relu)
        dummy_input = torch.randn(1, 3, 1024).cuda().float()
        with torch.no_grad():
            self.model(dummy_input)

    def process_data(self, data):
        try:
            if isinstance(data, dict) and data.get("msg_type") == "CONFIG":
                            self.width = data["width"]
                            self.height = data["height"]
                            if self.load_message:
                                return f"{self.load_message}\n相机参数初始化成功\n相机参数: {self.width}x{self.height}"
                            return "相机参数初始化成功"
            tensor_data = process_evs_numpy2tensor(
                data,
                src_width=self.width,
                src_height=self.height,
                dst_width=640,
                dst_height=480
            ).cuda(non_blocking=True).float()

            with torch.no_grad():
                output = self.model(tensor_data)
                result = output.squeeze().cpu().numpy().tolist() 
            res_text = f"输出结果为：{result}" 
            return res_text

        except Exception as e:
            error_msg = f"模型推理出错: {str(e)}"
            print(error_msg)
            return error_msg
