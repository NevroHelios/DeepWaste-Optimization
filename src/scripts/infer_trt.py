import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
from pathlib import Path
from typing import Literal

def infer_trt(mode: Literal['int8', 'fp16', 'int4'] = 'int8'):

    ENGINE_PATH = f"../../exports/model_{mode}_bs64.engine"  
    DATA_DIR = Path("../../data/CALIB")
    OUT_DIR = Path(f"../../data/TRT_OUT_{mode.upper()}")
    OUT_DIR.mkdir(exist_ok=True)

    assert os.path.exists(ENGINE_PATH), f"Engine file {ENGINE_PATH} does not exist."

    BATCH = 64
    NUM_CLASSES = 6  

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


    with open(ENGINE_PATH, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    input_shape = (BATCH, 3, 224, 224)
    output_shape = (BATCH, NUM_CLASSES)

    d_input = cuda.mem_alloc(int(np.prod(input_shape)) * 4)
    d_output = cuda.mem_alloc(int(np.prod(output_shape)) * 4)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    def infer(x_np):
        cuda.memcpy_htod_async(d_input, x_np, stream) # input -> device

        context.set_tensor_address(
            engine.get_tensor_name(0), int(d_input)
        )
        context.set_tensor_address(
            engine.get_tensor_name(1), int(d_output)
        )

        context.execute_async_v3(stream.handle)

        y = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(y, d_output, stream)
        stream.synchronize()

        return y

    for img_file in sorted(DATA_DIR.glob("test_images_*.npy")):
        idx = img_file.stem.split("_")[-1]
        x = np.load(img_file)
        y_pred = infer(x)

        np.save(OUT_DIR / f"trt_preds_{idx}.npy", y_pred)

    print("TensorRT inference complete.")

if __name__ == "__main__":
    # infer_trt(mode='int4')
    # infer_trt(mode='fp16')
    infer_trt(mode='int8')