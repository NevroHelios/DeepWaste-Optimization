
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

from src.models.pretrained import get_pretrained_model

# Configuration
BATCH_SIZE = 64
NUM_LOOPS = 100
WARMUP = 10
EXPORTS_DIR = Path("../../exports")
RESULTS_PATH = Path("../../benchmark_results.csv")
PLOT_PATH = Path("../../benchmark_plot.png")

def measure_throughput_latency(func, input_data, name):
    # Warmup
    print(f"[{name}] Warming up...")
    for _ in range(WARMUP):
        func(input_data)
    
    if torch and torch.cuda.is_available() and "PyTorch" in name:
        torch.cuda.synchronize()
    elif trt and "TensorRT" in name:
        cuda.Context.synchronize()

    # Benchmark
    print(f"[{name}] Benchmarking...")
    t0 = time.time()
    for _ in range(NUM_LOOPS):
        func(input_data)
    
    if torch and torch.cuda.is_available() and "PyTorch" in name:
        torch.cuda.synchronize()
    elif trt and "TensorRT" in name:
        cuda.Context.synchronize()
        
    t1 = time.time()

    total_time = t1 - t0
    total_samples = NUM_LOOPS * BATCH_SIZE
    
    latency_ms = (total_time / NUM_LOOPS) * 1000 # Latency per Batch
    
    throughput = total_samples / total_time
    
    print(f"[{name}] Latency: {latency_ms:.4f} ms/batch | Throughput: {throughput:.2f} img/sec")
    return latency_ms, throughput

def benchmark_pytorch(device="cuda"):
    print(f"\n--- PyTorch ({device.upper()}) ---")
    if not torch:
        print("PyTorch not installed.")
        return None, None
        
    model = get_pretrained_model(num_classes=6, freeze_strategy='finetune_all')
    model.to(device)
    model.eval()

    x = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
    
    def func(data):
        with torch.no_grad():
            model(data)

    return measure_throughput_latency(func, x, f"PyTorch {device.upper()}")

def benchmark_onnx(model_name, provider="CUDAExecutionProvider"):
    print(f"\n--- ONNX {model_name} ---")
    if not ort:
        print("ONNX Runtime not installed.")
        return None, None

    model_path = EXPORTS_DIR / f"{model_name}.onnx"
    if not model_path.exists():
        print(f"Skipping {model_name}, file not found.")
        return None, None

    try:
        sess = ort.InferenceSession(str(model_path), providers=[provider])
    except Exception as e:
        print(f"Failed to create ONNX session: {e}")
        return None, None

    input_name = sess.get_inputs()[0].name
    dtype = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
    x = np.random.randn(BATCH_SIZE, 3, 224, 224).astype(dtype)

    def func(data):
        sess.run(None, {input_name: data})

    res = measure_throughput_latency(func, x, f"ONNX {model_name}")
    return res

def benchmark_tensorrt(mode):
    print(f"\n--- TensorRT {mode} ---")
    if not trt:
        print("TensorRT not installed/importable.")
        return None, None

    engine_path = EXPORTS_DIR / f"model_{mode}_bs{BATCH_SIZE}.engine"
    if not engine_path.exists():
        print(f"Skipping TRT {mode}, file not found.")
        return None, None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    bindings = []
    inputs = []
    outputs = []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        
        vol = 1
        for s in shape: vol *= s
        
        np_dtype = np.float32
        if dtype == trt.DataType.HALF: np_dtype = np.float16
        elif dtype == trt.DataType.INT8: np_dtype = np.int8
        
        size = vol * np.dtype(np_dtype).itemsize
        d_ptr = cuda.mem_alloc(size)
        
        bindings.append(int(d_ptr))
        context.set_tensor_address(name, int(d_ptr))

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            dummy_input = np.random.randn(*shape).astype(np_dtype)
            inputs.append({'ptr': d_ptr, 'host': dummy_input})
        else:
            outputs.append(d_ptr)

    def func_simple(data):
        for inp in inputs:
            cuda.memcpy_htod_async(inp['ptr'], inp['host'], stream)
        context.execute_async_v3(stream.handle)
        stream.synchronize()

    return measure_throughput_latency(func_simple, None, f"TensorRT {mode}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-trt", action="store_true")
    args = parser.parse_args()

    results = []

    # 1. PyTorch
    if torch:
        l, t = benchmark_pytorch("cpu")
        if l: results.append({"Model": "PyTorch (CPU)", "Latency (ms)": l, "Throughput (img/s)": t})
        
        if torch.cuda.is_available():
            l, t = benchmark_pytorch("cuda")
            if l: results.append({"Model": "PyTorch (GPU)", "Latency (ms)": l, "Throughput (img/s)": t})

    # 2. ONNX
    if ort:
        l, t = benchmark_onnx("model_fp32")
        if l: results.append({"Model": "ONNX FP32", "Latency (ms)": l, "Throughput (img/s)": t})
        
        l, t = benchmark_onnx("model_fp16") # Assuming this exists
        if l: results.append({"Model": "ONNX FP16", "Latency (ms)": l, "Throughput (img/s)": t})

    # 3. TensorRT
    if not args.skip_trt and trt:
        l, t = benchmark_tensorrt("fp16")
        if l: results.append({"Model": "TensorRT FP16", "Latency (ms)": l, "Throughput (img/s)": t})
        
        l, t = benchmark_tensorrt("int8")
        if l: results.append({"Model": "TensorRT INT8", "Latency (ms)": l, "Throughput (img/s)": t})
    elif args.skip_trt:
        # Inject Hardcoded results from User if skipping
        print("Injecting hardcoded TRT results from previous runs...")
        # FP16: 9.26 ms, 6905 img/s
        results.append({"Model": "TensorRT FP16", "Latency (ms)": 9.27, "Throughput (img/s)": 6905.0})
        # INT8: 9.44 ms, 6777 img/s
        results.append({"Model": "TensorRT INT8", "Latency (ms)": 9.44, "Throughput (img/s)": 6778.0})

    # Save
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(RESULTS_PATH, index=False)
        print("\nResults saved to", RESULTS_PATH)
        print(df)

        # Plot
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(data=df, x="Model", y="Throughput (img/s)", hue="Model")
        plt.title("Inference Throughput Comparison (Batch Size 64)")
        plt.ylabel("Images / Second")
        plt.xticks(rotation=45)
        
        # Add values on top
        for p in barplot.patches:
            if p.get_height() > 0:
                barplot.annotate(f'{int(p.get_height())}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 9), 
                               textcoords = 'offset points')

        plt.tight_layout()
        plt.savefig(PLOT_PATH)
        print("Plot saved to", PLOT_PATH)
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()
