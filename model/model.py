import os
import time
import torch
import numpy as np
from PIL import Image
import tensorrt as trt
import ctypes
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from .utils import RealESRGANer

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'weights\RealESRGAN_x4plus.pth')
ONNX_PATH = os.path.join(ROOT_DIR, 'optimizations\esrgan.onnx')
ENGINE_PATH = os.path.join(ROOT_DIR, 'optimizations\esrgan.trt')
WEIGHT_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

model_storage = {}

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class UnoptimizedModel():
   def __init__(self, netscale=4, useGPU=False):
      model_path = MODEL_PATH
      # Download weights in case they don't exist
      if not os.path.isfile(MODEL_PATH):
         model_path = load_file_from_url(url=WEIGHT_URL, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

      device = 'cpu'
      if useGPU and torch.cuda.is_available():
         device = 'cuda:0'

      network_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
      model = RealESRGANer(
         scale=netscale,
         model_path=model_path,
         dni_weight=None,
         model=network_model,
         tile=0,
         tile_pad=10,
         pre_pad=0,
         half=useGPU,
         device=device)
      
      self.model = model

   def enhance(self, img, outscale: int):
      start = time.time()
      imgData = np.asarray(img, dtype='int32')
      image, _ = self.model.enhance(imgData, outscale)
      return image, time.time() - start

class CPUUnoptimizedModel(UnoptimizedModel, metaclass=SingletonMeta):
   def __init__(self, netscale=4):
      super().__init__(netscale, False)

class GPUUnoptimizedModel(UnoptimizedModel, metaclass=SingletonMeta):
   def __init__(self, netscale=4):
      super().__init__(netscale, True)

class OptimizedModel(metaclass=SingletonMeta):
   def __init__(self, netscale=4):
      self.netscale = netscale
      self.engine = None
      self.context = None
      self.input_ptr = None
      self.output_ptr = None
      
      start = time.time()
      model_path = MODEL_PATH
      # Download weights in case they don't exist
      if not os.path.isfile(MODEL_PATH):
         model_path = load_file_from_url(url=WEIGHT_URL, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

      model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)

      if (not os.path.isfile(ENGINE_PATH)):
         if not os.path.isfile(ONNX_PATH):
            print("Perf log: Exporting ONNX file")
            model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
            # set the train mode to false since we will only run the forward pass.
            model.train(False)
            model.eval()
            dummy_input = torch.randn(1, 3, 256, 256)
            torch.onnx.export(
               model,
               dummy_input,
               ONNX_PATH,
               opset_version=13,
               input_names=['input'],
               output_names=['output']
            )
            print(f'Perf log ONNX export: {time.time() - start:.2f} seconds')


         # Turn into TensorRT model
         print("Perf log: Exporting TensorRT file")
         builder = trt.Builder(TRT_LOGGER)
         network = builder.create_network(0)
         parser = trt.OnnxParser(network, TRT_LOGGER)
         success = parser.parse_from_file(ONNX_PATH)

         for idx in range(parser.num_errors):
            print(parser.get_error(idx))

         if success:
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
            serialized_engine = builder.build_serialized_network(network, config)

            with open(ENGINE_PATH, 'wb') as f:
               f.write(serialized_engine)

         print(f'Perf log TensorRT export: {time.time() - start:.2f} seconds')

      self.runtime = trt.Runtime(TRT_LOGGER)

   def pad_image(self, img_np, multiple=256):
      _, _, h, w = img_np.shape
      pad_h = (multiple - h % multiple) % multiple
      pad_w = (multiple - w % multiple) % multiple

      pad = (
         (0,0),
         (0,0),
         (0, pad_h),
         (0, pad_w)
      )

      padded_img = np.pad(img_np, pad_width=pad, mode='constant', constant_values=0)
      return padded_img, pad_h, pad_w
   
   def unpad_image(self, img_np, pad_h, pad_w):
      if pad_h == 0 or pad_w == 0:
         return img_np
      
      _, _, h, w = img_np.shape
      new_h = h - pad_h * self.netscale
      new_w = w - pad_w * self.netscale
      return img_np[:, :, :new_h, :new_w]

   def enhance(self, img, outscale: int): 
      start_inference = time.time()
      height, width = img.size
      # Reshape into (256, 256, 3)
      resizedImage = img.resize((256,256)).convert("RGB")
      # Divide channels into 255
      resizedData = np.array(resizedImage).astype(np.float32) / 255.
      # Add Batch dimension (N, C, H, W)
      resizedData = resizedData.transpose(2, 0, 1)[None, :, :, :]

      #Add padding
      paddedData, pad_h, pad_w = self.pad_image(resizedData)

      #load engine
      if self.engine is None:
         with open(ENGINE_PATH, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

      # Create context
      if self.context is None:
         self.context = self.engine.create_execution_context()

      # Get tensor names, shapes and types
      input_name = self.engine.get_tensor_name(0)
      output_name = self.engine.get_tensor_name(1)
      input_shape = self.engine.get_tensor_shape(input_name)
      output_shape = self.engine.get_tensor_shape(output_name)
      output_dtype = trt.nptype(self.engine.get_tensor_dtype(output_name))

      # Set input shape and create mimic arrays
      self.context.set_input_shape(input_name, input_shape)
      result = np.empty(output_shape, dtype=output_dtype)

      # Allocate memmory
      cudart = ctypes.WinDLL('cudart64_110.dll')
      if self.input_ptr is None:
         self.input_ptr = ctypes.c_void_p()
         cudart.cudaMalloc(ctypes.byref(self.input_ptr), paddedData.nbytes)
      
      if self.output_ptr is None:
         self.output_ptr = ctypes.c_void_p()
         cudart.cudaMalloc(ctypes.byref(self.output_ptr), result.nbytes)

      # set tensor references
      self.context.set_tensor_address(input_name, self.input_ptr.value)
      self.context.set_tensor_address(output_name, self.output_ptr.value)

      # transfer image to GPU
      cudart.cudaMemcpy(self.input_ptr, paddedData.ctypes.data_as(ctypes.c_void_p), paddedData.nbytes, 1)
      # run inference
      self.context.execute_v2([self.input_ptr.value, self.output_ptr.value])

      """
         The following lines cause a bottleneck with the batch is only 1.
         The bottleneck happens as the copying the buffer from the GPU to the CPU is needed
         in order to return the resulting image and to render it in the UI. So even if the
         inference takes about 200ms, moving the image takes another 300ms which makes the whole
         operation take about half a second
      """
      cudart.cudaMemcpy(result.ctypes.data_as(ctypes.c_void_p), self.output_ptr, result.nbytes, 2)

      # post process image
      unpaddedData = self.unpad_image(result, pad_h, pad_w)
      output = unpaddedData[0].transpose(1, 2, 0) * 255
      outputImage = Image.fromarray(output.clip(0, 255).astype(np.uint8))
      inference_time = time.time() - start_inference
      return outputImage.resize((height * outscale, width * outscale)), inference_time