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

def process_evs_numpy2tensor(data_numpy,weight = 640,height = 480 ,sample_size = 1024):
    """处理numpy的数据,输出tensor数据,中间会对数据进行sort分类,以及归一化"""

    # 归一化
    x_values = data_numpy[:, 0] / weight
    y_values = data_numpy[:, 1] / height
    t_values = data_numpy[:, 2]
    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)

    # 选取1024个点
    current_sample_size = len(t_values)
    indices = np.random.choice(current_sample_size, sample_size, replace=False)
    indices = np.sort(indices)
    x = x_values[indices]
    y = y_values[indices]
    t = t_values[indices]
    data_chulihou  = np.stack((t,x,y), axis=-1) #修改
    data_tensor = torch.from_numpy(data_chulihou).unsqueeze(0).permute(0, 2, 1)
    return data_tensor

class EventMambaPredictor:
    def __init__(self):
        self.width = 640 
        self.height = 480
        self.frame = 0  #删除
        self.model = EventMamba(num_classes=2).cuda()
        #加载权重
        self.model.eval()
        self.model.apply(inplace_relu)

    def process_data(self, data):
        try:
            self.frame += 1 #删除
            if isinstance(data, dict) and data.get("msg_type") == "CONFIG":
                            self.cam_width = data["width"]
                            self.cam_height = data["height"]
                            print(f"收到相机参数更新: {self.cam_width}x{self.cam_height}")
                            return "相机参数初始化成功" # 直接返回，不需要推理

            tensor_data = process_evs_numpy2tensor(data).cuda() #(1, 3, 1024)

            with torch.no_grad():
                output = self.model(tensor_data)
                result = output.squeeze().cpu().numpy().tolist()
            
            res_text = f"输出结果为：{result}" 
            return res_text

        except Exception as e:
            error_msg = f"模型推理出错: {str(e)}"
            print(error_msg)
            return error_msg