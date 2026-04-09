import torch
import numpy as np
import os
import sys
from backend.Eventmamba.models.eventmamba_v1 import EventMamba
current_root = os.getcwd()
modules_parent_dir = os.path.join(current_root, "backend", "Eventmamba", "models")

if modules_parent_dir not in sys.path:
    sys.path.append(modules_parent_dir)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


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

            data_tensor = torch.from_numpy(data).unsqueeze(0).permute(0, 2, 1).cuda().float()

            with torch.no_grad():
                output = self.model(data_tensor)
                result = output.squeeze().cpu().numpy().tolist()
            res_text = f"输出结果为：{result}|cropped:True"
            return res_text

        except Exception as e:
            error_msg = f"模型推理出错: {str(e)}"
            print(error_msg)
            return error_msg
