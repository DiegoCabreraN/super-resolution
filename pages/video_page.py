import cv2
import time
import numpy as np
import streamlit as st
import imageio
from PIL import Image
import pynvml
from model.model import GPUUnoptimizedModel, CPUUnoptimizedModel, OptimizedModel


pynvml.nvmlInit()
deviceCount = pynvml.nvmlDeviceGetCount()

st.markdown("""
   # Video Upsampling
   > This project uses the Real ESRGAN model to process the video. Check the different implementations with the buttons to validate the original behavior, the original behavior with the GPU and the TensorRT behavior
""")

uploaded_file = st.file_uploader("Choose a video to upscale", type=["mp4"])

if uploaded_file is not None:

   col1, col2 = st.columns([1,1])
   video_bytes = uploaded_file.read()


   model = None

   break_generation = False

   button_col1, button_col2, button_col3 = st.columns([1,1,1])
   with button_col1:
      if st.button('Generate with CPU'):
         model = CPUUnoptimizedModel(4)
         break_generation = False
   with button_col2:
      if st.button('Generate with GPU'):
         model = GPUUnoptimizedModel(4)
         break_generation = False
   with button_col3:
      if st.button('Generate with TensorRT'):
         model = OptimizedModel(4)
         break_generation = False
   if st.button('Cancel', type='secondary'):
      break_generation = True



   with col1:
      st.video(video_bytes, 'video/mp4', loop=True, muted=True)
      st.caption('Original video')
   with col2:
      with open('resources/tmp.mp4', 'wb') as tmp:
         tmp.write(video_bytes)

      if model is not None:
         video = cv2.VideoCapture('resources/tmp.mp4')


         last_fps_update = time.time()
         fps_count = 0

         video_placeholder = st.empty()
         inference_caption = st.empty()
         video_caption = st.empty()

         generated_frames = []
         generation_time = time.time()

         while video.isOpened():
            success, image = video.read()

            current_frame = None

            if success:
               color_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
               current_frame, inference_time = model.enhance(color_image, 4)
               generated_frames.append(current_frame)
               fps_count += 1
               video_placeholder.image(current_frame, use_container_width=True)
               inference_caption.caption(f'- Inference time: {inference_time:.2f}')
               if break_generation:
                  break
            else:
               break

            if time.time() - last_fps_update > 10:
               elapsed_time = time.time() - last_fps_update
               last_fps_update = time.time()

               handle = pynvml.nvmlDeviceGetHandleByIndex(0)
               util = pynvml.nvmlDeviceGetUtilizationRates(handle)
               video_caption.markdown(f"""- {fps_count/elapsed_time:.2f} Frames per second\n- GPU Utilization rate: {util.gpu/100.0:3.1%}\n- GPU Memory rate: {util.memory/100.0:3.1%}""")
               fps_count = 0

         cv2.destroyAllWindows()
         video.release()


         writer = imageio.get_writer('resources/output.mp4', format='FFMPEG', mode='I', fps=24)

         for j in range(len(generated_frames)):
            writer.append_data(np.array(generated_frames[j]))
            if j == len (generated_frames) - 1:
               writer.close()
               inference_caption.empty()
               video_placeholder.video('resources/output.mp4', 'video/mp4')
               video_caption.caption(f'Generated video (took {time.time() - generation_time:.2f}s)')

   # 

   # cv2.destroyAllWindows()
   # video.release()
   # Load image as np array
   # img = Image.open(uploaded_file)
   # img.load()

   # # render old image
   # with col1:
   #    st.image(img, use_container_width=True)
   #    st.caption('original image')

   
   # # Generate images with the model
   # try:
   #    scale = 4
      
   #    start = time.time()

   #    model = None
      
   #    button_col1, button_col2, button_col3 = st.columns([1,1,1])

   #    with button_col1:
   #       if st.button("Generate with CPU"):
   #          model = UnoptimizedModel(scale)

   #    with button_col2:
   #       if st.button("Generate with GPU"):
   #          model = UnoptimizedModel(scale, True)

   #    with button_col3:
   #       if st.button("Generate with Optimized GPU"):
   #          model = OptimizedModel(scale)
   #          pass

   #    if model is not None:
   #       generated_image = model.enhance(img, outscale=scale)

   #       end = time.time()
   #       with col2:
   #          st.image(generated_image, use_container_width=True)
   #          handle = pynvml.nvmlDeviceGetHandleByIndex(0)
   #          util = pynvml.nvmlDeviceGetUtilizationRates(handle)
   #          st.caption(f'Image generated in {end - start:.2f} seconds')
   #          st.caption(f'GPU Utilization rate: {util.gpu/100.0:3.1%}')
   #          st.caption(f'GPU Memory rate: {util.memory/100.0:3.1%}')


   # except RuntimeError as error:
   #    print('Error', error)
   #    print('If you encounter CUDA out of memory, try to set tile with a smaller number.')
