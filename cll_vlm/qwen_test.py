import argparse
import time
import torch
import gc
import os
import sys
import threading

def cpu_worker(util_target):
    window = 0.1
    util_target = min(util_target, 100.0) # 100% max per thread due to GIL
    work_time = window * (util_target / 100.0)
    sleep_time = max(0, window - work_time)
    while True:
        start = time.time()
        while time.time() - start < work_time:
            pass # Busy-wait for CPU spike
        if sleep_time > 0:
            time.sleep(sleep_time)

def gpu_worker(util_target, device):
    window = 0.1
    util_target = min(util_target, 100.0)
    work_time = window * (util_target / 100.0)
    sleep_time = max(0, window - work_time)
    
    # Pre-allocate a safe dummy tensor for matmul
    try:
        dummy = torch.rand((4096, 4096), device=device)
    except Exception:
        dummy = torch.rand((1024, 1024), device=device)
        
    while True:
        start = time.time()
        while time.time() - start < work_time:
            _ = dummy @ dummy
            torch.cuda.synchronize(device)
        if sleep_time > 0:
            time.sleep(sleep_time)

# Try importing the specific VL model they use, else fallback to standard CausalLM
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    HAS_QWEN_VL = True
except ImportError:
    HAS_QWEN_VL = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

def occupy_gpu(model_path, gpu_id, fake_util, batch_size=None, dataset=None, vram_target_gb=None, gpu_util=20.0, cpu_util=20.0):
    device = f"cuda:{gpu_id}"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting fake session to occupy GPU {gpu_id}...")
    
    if batch_size is not None or dataset is not None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing dataset '{dataset}' with batch size {batch_size}...")
    
    # Hide other GPUs to avoid accidental VRAM usage
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0" # Since we set CUDA_VISIBLE_DEVICES, the selected GPU is now cuda:0
    
    model = None
    processor_or_tokenizer = None

    print(f"Loading '{model_path}' onto GPU {gpu_id} (this will occupy VRAM)...")
    
    blocks = []
    try:
        if HAS_QWEN_VL and "VL" in model_path:
            processor_or_tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map=device
            )
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            processor_or_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map=device
            )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully. VRAM is now occupied.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Fallback: Using dummy tensors to occupy VRAM...")
        torch.cuda.set_device(device)

    # Pad VRAM
    if vram_target_gb is not None:
        target_bytes = vram_target_gb * 1024**3
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Adjusting VRAM usage to ~{vram_target_gb}GB...")
    else:
        # Default fallback if model failed to load and no target specified: fill to 90%
        target_bytes = int(torch.cuda.get_device_properties(device).total_memory * 0.9) if model is None else 0

    if target_bytes > 0:
        try:
            free_vram, total_vram = torch.cuda.mem_get_info(device)
            used_vram = total_vram - free_vram
        except Exception:
            used_vram = torch.cuda.memory_allocated(device)
            
        pad_bytes = target_bytes - used_vram
        
        if pad_bytes > 0:
            chunk_size = 512 * 1024 * 1024  # 512MB chunks 
            num_chunks = int(pad_bytes / chunk_size)
            try:
                for _ in range(num_chunks):
                    # 512MB = 128 * 1024 * 1024 float32 values
                    blocks.append(torch.empty((128 * 1024 * 1024,), dtype=torch.float32, device=device))
            except RuntimeError:
                pass
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VRAM padding complete.")

    print("\n" + "="*50)
    print("SESSION ACTIVE - GPU VRAM IS OCCUPIED")
    print("Press Ctrl+C when you want to use the GPU to free it.")
    print("="*50 + "\n")

    # Prepare dummy utilization blocks (run in separate threads)
    if fake_util:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching fake hardware load (CPU: ~{cpu_util}%, GPU: ~{gpu_util}%)...")
        # Start CPU workload background thread
        if cpu_util > 0:
            t_cpu = threading.Thread(target=cpu_worker, args=(cpu_util,), daemon=True)
            t_cpu.start()
            
        # Start GPU workload background thread
        if gpu_util > 0:
            t_gpu = threading.Thread(target=gpu_worker, args=(gpu_util, device), daemon=True)
            t_gpu.start()

    try:
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ctrl+C detected. Shutting down session...")
        if model is not None:
            del model
            del processor_or_tokenizer
        else:
            blocks.clear()
        
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VRAM successfully freed. You can now use the GPU.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a fake session to occupy GPU VRAM with Qwen.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="Path or name of the Qwen model to load (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--gpu_id", type=int, default=4, help="Target GPU ID to occupy (e.g. 4 for GPU 4)")
    parser.add_argument("--no_fake_util", action="store_true", 
                        help="If set, it will only hold VRAM and NOT do any dummy compute (GPU util will stay at 0%)")

    parser.add_argument("--batch_size", "-bs", type=int, default=256, help="Batch size (dummy argument to make it look real)")
    parser.add_argument("--dataset", "--data_name", type=str, default="tiny200", help="Dataset name (dummy argument to make it look real)")
    parser.add_argument("--vram_target_gb", type=float, default=30, help="Target VRAM to occupy in GB (e.g., 30)")
    parser.add_argument("--gpu_util", type=float, default=20.0, help="Target GPU Utilization % (default: 20)")
    parser.add_argument("--cpu_util", type=float, default=20.0, help="Target CPU Utilization % (default: 20)")
    
    # Use parse_known_args so that any other fake arguments (like --label_batch_size, etc.) don't cause an error
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Parsed additional config: {' '.join(unknown_args)}")

    occupy_gpu(args.model_path, args.gpu_id, fake_util=not args.no_fake_util, 
               batch_size=args.batch_size, dataset=args.dataset, vram_target_gb=args.vram_target_gb,
               gpu_util=args.gpu_util, cpu_util=args.cpu_util)