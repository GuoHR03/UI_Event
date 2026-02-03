#这个是切换的表达方式

import numpy as np  # 必须导入 numpy 用于处理数组维度
import time
import os
import argparse

# Metavision SDK 导入
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Metavision Live Recorder with Dynamic Palette',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-o', '--output-dir', default="", help="Directory where to create RAW file")
    args = parser.parse_args()
    return args

def main():
    """ 主函数 """
    args = parse_args()

    # 1. 初始化相机 (HAL Device)
    try:
        device = initiate_device("")
        print("相机启动成功。")
    except OSError as e:
        print(f"无法打开相机: {e}")
        return

    # 2. 自动开始录制 (文件名带时间戳)
    if device.get_i_events_stream():
        filename = "recording_" + time.strftime("%y%m%d_%H%M%S", time.localtime()) + ".raw"
        log_path = filename
        if args.output_dir != "":
            log_path = os.path.join(args.output_dir, filename)
        
        print(f'正在录制到文件: {log_path}')
        device.get_i_events_stream().log_raw_data(log_path)

    # 3. 创建迭代器
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()

    # 4. 创建显示窗口 (模式设置为 BGR，即 3 通道)
    with MTWindow(title="Metavision Viewer (Press 'P' to switch color)", 
                  width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        
        # --- 颜色切换逻辑准备 ---
        available_palettes = [
            ColorPalette.Dark,      # 黑底彩色 (3通道)
            ColorPalette.Light,     # 白底彩色 (3通道)
            ColorPalette.Gray,      # 纯灰度 (2通道 -> 需要特殊处理)
            ColorPalette.CoolWarm   # 冷暖色 (3通道)
        ]
        current_palette_idx = 0

        # 初始化算法 (默认 Dark)
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width, sensor_height=height, fps=25,
            palette=available_palettes[current_palette_idx])

        # --- 👇👇👇 核心修复：显示回调函数 👇👇👇 ---
        def on_cd_frame_cb(ts, cd_frame):
            # 检查图像维度
            # 如果是二维数组 (Gray模式产生的是 Height x Width)，窗口会报错
            if cd_frame.ndim == 2:
                # 使用 numpy 将其堆叠为 3 通道 (H, W, 3)
                cd_frame = np.stack((cd_frame,)*3, axis=-1)
            
            # 现在无论是彩色还是灰度，都是 3 维数据了，可以安全显示
            window.show_async(cd_frame)
        # ---------------------------------------------

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # --- 键盘控制逻辑 ---
        def keyboard_cb(key, scancode, action, mods):
            nonlocal event_frame_gen, current_palette_idx

            if action != UIAction.PRESS:
                return

            # 退出键
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            
            # 'P' 键切换颜色
            elif key == UIKeyEvent.KEY_P:
                # 1. 更新索引
                current_palette_idx = (current_palette_idx + 1) % len(available_palettes)
                new_palette = available_palettes[current_palette_idx]
                print(f"-> 切换模式: {new_palette}")

                # 2. 重建算法对象
                event_frame_gen = PeriodicFrameGenerationAlgorithm(
                    sensor_width=width, sensor_height=height, fps=25,
                    palette=new_palette)
                
                # 3. 【必须】重新绑定回调函数
                event_frame_gen.set_output_callback(on_cd_frame_cb)

        window.set_keyboard_callback(keyboard_cb)

        print("程序运行中... 按 'P' 切换颜色，按 'Q' 退出。")

        # 5. 主循环
        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                if device.get_i_events_stream():
                    device.get_i_events_stream().stop_log_raw_data()
                print("录制结束，程序退出。")
                break

if __name__ == "__main__":
    main()