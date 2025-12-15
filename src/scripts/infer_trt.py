import os
from pathlib import Path
from typing import Literal

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


def infer_trt(
    mode: Literal["int8", "fp16", "int4"] = "int8",
    BATCH: int = 64,
    NUM_CLASSES: int = 6,
):
    import pycuda.autoinit  # Required for CUDA context

    ENGINE_PATH = f"../../exports/model_{mode}_bs{BATCH}.engine"
    DATA_DIR = Path("../../data/CALIB")
    OUT_DIR = Path(f"../../data/TRT_OUT_{mode.upper()}")
    OUT_DIR.mkdir(exist_ok=True)

    assert os.path.exists(ENGINE_PATH), f"Engine file {ENGINE_PATH} does not exist."

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(ENGINE_PATH, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Dynamic Tensor Inspection
    tensor_map = {}  # name -> {'mode': 'input'/'output', 'dtype': np.dtype, 'shape': ..., 'ptr': cuda_ptr}
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode_ = engine.get_tensor_mode(name) # trt.TensorIOMode.INPUT or OUTPUT
        dtype_ = engine.get_tensor_dtype(name) # trt.DataType.FLOAT, HALF, etc.
        shape = engine.get_tensor_shape(name) # (BATCH, C, H, W)
        
        # Resolve dynamic batch size if needed (assuming fixed BATCH for now or matching engine)
        if -1 in shape:
             # If engine has dynamic shapes, you might need context.set_input_shape here
             pass

        # Map TRT dtype to Numpy dtype
        np_dtype = np.float32
        if dtype_ == trt.DataType.HALF:
            np_dtype = np.float16
        elif dtype_ == trt.DataType.INT8:
            np_dtype = np.int8
        
        # Allocate CUDA memory
        vol = 1
        for dim in shape:
            vol *= abs(dim) # Handle -1 safely if present, though alloc needs real size
        
        nbytes = vol * np.dtype(np_dtype).itemsize
        d_ptr = cuda.mem_alloc(nbytes)
        
        tensor_map[name] = {
            "mode": mode_,
            "dtype": np_dtype,
            "shape": shape,
            "ptr": d_ptr
        }
        
        # Bind address
        context.set_tensor_address(name, int(d_ptr))

    stream = cuda.Stream()

    def infer(x_np):
        # find the input tensor(s)
        for name, info in tensor_map.items():
            if info['mode'] == trt.TensorIOMode.INPUT:
                # Ensure input data matches engine expectation
                if x_np.dtype != info['dtype']:
                    x_np = x_np.astype(info['dtype'])
                
                # Copy Host -> Device
                cuda.memcpy_htod_async(info['ptr'], x_np, stream)

        # Execute
        context.execute_async_v3(stream.handle)

        # Copy Device -> Host
        outputs = {}
        for name, info in tensor_map.items():
            if info['mode'] == trt.TensorIOMode.OUTPUT:
                host_mem = np.empty(info['shape'], dtype=info['dtype'])
                cuda.memcpy_dtoh_async(host_mem, info['ptr'], stream)
                outputs[name] = host_mem

        stream.synchronize()
        
        # Return the first output (assuming single output for classification)
        return list(outputs.values())[0]

    for img_file in sorted(DATA_DIR.glob("test_images_*.npy")):
        idx = img_file.stem.split("_")[-1]
        x = np.load(img_file)
        
        # Basic shape check/broadcast if x is single image but batch is 64
        # (Assuming x is (BATCH, ...) or adapting logic as needed)
        
        y_pred = infer(x)
        np.save(OUT_DIR / f"trt_preds_{idx}.npy", y_pred)

    print("TensorRT inference complete.")

    # Benchmark
    print(f"\nBenchmarking TensorRT {mode} (Batch Size {BATCH})...")
    import time
    
    # Prepare Dummy Input
    input_dtype = np.float32
    for name, info in tensor_map.items():
         if info['mode'] == trt.TensorIOMode.INPUT:
             input_dtype = info['dtype']
             break

    dummy_input = np.random.randn(BATCH, 3, 224, 224).astype(input_dtype)

    # Warmup
    for _ in range(20):
        infer(dummy_input)

    # Measure
    t0 = time.time()
    n_loops = 200
    for _ in range(n_loops):
        infer(dummy_input)
    t1 = time.time()
    
    total_time = t1 - t0
    avg_latency = (total_time / n_loops) * 1000  # ms per batch
    throughput = (n_loops * BATCH) / total_time  # images per second
    
    print(f"tensorrt {mode}")
    print(f"Latency: {avg_latency:.4f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")


if __name__ == "__main__":
    # infer_trt(mode='int4')
    infer_trt(mode='fp16')
    infer_trt(mode="int8")
