import collections
import pycuda.driver as cuda
import tensorrt as trt
from collections import OrderedDict
import numpy as np


class TRTInferenceObject:
    """
    Class for performing inference using a TensorRT engine with RTDETRv2 models for player and ball detection.
    
    Based on:
    - https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/references/deploy/rtdetrv2_tensorrt.py
    - https://github.com/NVIDIA/TensorRT/blob/97ff24489d0ea979c418c7a0847dfc14c8483846/quickstart/IntroNotebooks/onnx_helper.py
    """
    def __init__(self, engine_path):
        
        self.engine_path = engine_path
        self.load(engine_path)
        self.allocate_memory()
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

        self.stream = cuda.Stream()
        
    def load(self, engine_path):
        print('Loading TensorRT engine from {:}'.format(engine_path))
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError('Failed to deserialize TensorRT engine')
        self.context = self.engine.create_execution_context()
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names
    
    def allocate_memory(self):
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'gpu_mem'))
        bindings = OrderedDict()
        for i, name in enumerate(self.engine):
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            data = np.empty(shape).astype(dtype)
            gpu_mem = cuda.mem_alloc(data.nbytes)
            bindings[name] = Binding(name, dtype, shape, data, gpu_mem) 
            self.context.set_tensor_address(name, int(gpu_mem))
            
        self.bindings = bindings
        self.stream = cuda.Stream()
        
    def predict(self, blob):
        # Transfer input data to device
        for n in self.input_names:
            cuda.memcpy_htod_async(self.bindings[n].gpu_mem, blob[n], self.stream)
        
        # Execute model
        self.context.execute_async_v3(self.stream.handle)
        
        # Transfer predictions back
        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].gpu_mem, self.stream)
            outputs[n] = self.bindings[n].data
        
        self.stream.synchronize()
        
        return outputs
    
    
class TRTInferenceKeypoint:
    """
    Class for performing inference using a TensorRT engine with the keypoint detection model.
    
    Based on:
    - https://github.com/NVIDIA/TensorRT/blob/97ff24489d0ea979c418c7a0847dfc14c8483846/quickstart/IntroNotebooks/onnx_helper.py
    """
    def __init__(self, engine_path, num_keypoints, target_dtype= np.float32):
        
        self.num_keypoints = num_keypoints
        self.target_dtype = target_dtype
        self.load(engine_path)
        self.stream = None
        
    def load(self, engine_path):
        print('Loading TensorRT engine from {:}'.format(engine_path))
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError('Failed to deserialize TensorRT engine')
        self.context = self.engine.create_execution_context()

    def allocate_memory(self, batch):
        self.output = np.empty(self.num_keypoints * 2, dtype = self.target_dtype) #TODO (?) Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(batch.nbytes)
        self.d_output = cuda.mem_alloc(self.output.nbytes)
        
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        assert(len(tensor_names) == 2)

        self.context.set_tensor_address(tensor_names[0], int(self.d_input))
        self.context.set_tensor_address(tensor_names[1], int(self.d_output))

        self.stream = cuda.Stream()    
        
    def predict(self, batch): 
        if self.stream is None:
            self.allocate_memory(batch)
            
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v3(self.stream.handle)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()
        
        return self.output