import subprocess
import time
import os
import sys
import multiprocessing
import numpy as np
try:
    import pynvml
except ImportError:
    print("请安装: pip install nvidia-ml-py")
    sys.exit(1)

# ----------------配置区域----------------
# 待测试的 id 列表 (对应您图片中的 0 到 6)
test_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# 基础路径配置
source_base = "/extdatashare/dir_baonerf/data_pro_c4_ours/dataset/huanong4/idout"
model_base = "/extdatashare/dir_baonerf/data_pro_c4_ours/gsout/huanong4"
# 训练脚本路径 (假设就在当前目录)
train_script = "train.py"
# 额外参数
extra_args = ["--eval", "--position_lr_init", "0.00064", "--detects", "mask", "-d", "depth", "--port", "1070"]
# ---------------------------------------

def get_current_gpu_usage(handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024 / 1024

def gpu_monitor_task(stop_event, stats_dict, device_id=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    baseline = get_current_gpu_usage(handle)
    stats_dict['baseline'] = baseline
    max_absolute = baseline
    
    while not stop_event.is_set():
        try:
            current = get_current_gpu_usage(handle)
            if current > max_absolute:
                max_absolute = current
                stats_dict['peak_absolute'] = max_absolute
        except:
            pass
        time.sleep(0.1) 
    pynvml.nvmlShutdown()

def run_single_gs_task(task_id):
    source_path = f"{source_base}/{task_id}"
    model_path = f"{model_base}/{task_id}"
    
    # 构造命令: python train.py -s ... -m ... --eval ...
    cmd = [sys.executable, train_script, "-s", source_path, "-m", model_path] + extra_args
    
    manager = multiprocessing.Manager()
    stats_dict = manager.dict({'baseline': 0.0, 'peak_absolute': 0.0})
    stop_event = multiprocessing.Event()
    
    monitor_p = multiprocessing.Process(target=gpu_monitor_task, args=(stop_event, stats_dict))
    monitor_p.start()
    time.sleep(2) # 等待基准稳定
    
    start_time = time.time()
    try:
        print(f"\n▶️ 正在运行 3DGS 任务 ID: {task_id}")
        # 运行训练并实时打印输出
        process = subprocess.Popen(cmd)
        process.wait()
    finally:
        end_time = time.time()
        stop_event.set()
        monitor_p.join()

    duration = end_time - start_time
    net_peak = stats_dict['peak_absolute'] - stats_dict['baseline']
    
    return duration, net_peak

if __name__ == "__main__":
    all_times = []
    all_mems = []
    
    log_file = "3dgs_benchmark_summary.txt"
    
    with open(log_file, "w", encoding='utf-8') as f:
        f.write(f"3DGS 批量运行实验报告 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")

    for tid in test_ids:
        duration, mem = run_single_gs_task(tid)
        all_times.append(duration)
        all_mems.append(mem)
        
        res_str = f"ID {tid}: 耗时 = {duration:.2f}s, 显存净增峰值 = {mem:.2f}MB"
        print(f"✅ {res_str}")
        
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(res_str + "\n")

    # 计算平均值
    avg_time = np.mean(all_times)
    avg_mem = np.mean(all_mems)
    
    summary_str = (
        f"\n" + "="*60 + "\n"
        f"统计摘要 (共 {len(test_ids)} 组实验):\n"
        f"平均总运行耗时: {avg_time:.2f} 秒\n"
        f"平均显存净增峰值: {avg_mem:.2f} MB\n"
        f"最大显存峰值记录: {np.max(all_mems):.2f} MB\n"
        + "="*60
    )
    
    print(summary_str)
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(summary_str)