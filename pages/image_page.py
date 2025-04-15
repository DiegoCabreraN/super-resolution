import time
import numpy as np
import streamlit as st
from PIL import Image
import pynvml
from model.model import GPUUnoptimizedModel, CPUUnoptimizedModel, OptimizedModel


pynvml.nvmlInit()
deviceCount = pynvml.nvmlDeviceGetCount()

st.markdown("""
   # Image Upsampling
   > This project uses the Real ESRGAN model to process the images. Check the different implementations with the buttons to validate the original behavior, the original behavior with the GPU and the TensorRT behavior
""")

uploaded_file = st.file_uploader("Choose an image to upscale", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   # Load image as np array
   img = Image.open(uploaded_file)
   img.load()
   col1, col2 = st.columns([1,1])

   # render old image
   with col1:
      st.image(img, use_container_width=True)
      st.caption('original image')

   
   # Generate images with the model
   try:
      scale = 4

      model = None
      
      button_col1, button_col2, button_col3 = st.columns([1,1,1])

      with button_col1:
         if st.button("Generate with CPU"):
            model = CPUUnoptimizedModel(scale)

      with button_col2:
         if st.button("Generate with GPU"):
            model = GPUUnoptimizedModel(scale)

      with button_col3:
         if st.button("Generate with TensorRT"):
            model = OptimizedModel(scale)
            pass

      if model is not None:
         generated_image, inference_time = model.enhance(img, outscale=scale)

         with col2:
            st.image(generated_image, use_container_width=True)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            st.caption(f'Image generated in {inference_time:.2f} seconds')
            st.caption(f'GPU Utilization rate: {util.gpu/100.0:3.1%}')
            st.caption(f'GPU Memory rate: {util.memory/100.0:3.1%}')


   except RuntimeError as error:
      print('Error', error)
      print('If you encounter CUDA out of memory, try to set tile with a smaller number.')
