<p align="center">
  <img src="resources/logo.png" height=120>
</p>

Super Resolution is a Streamlit application for Image/Video restoration. We extend the [Real ESRGAN model](https://github.com/xinntao/Real-ESRGAN) and optimize it by using NVidia's TensorRT.

### Demo Video
- 

---

## Dependencies and Installation
- This project was built in a Windows environment, and it uses the Windows Cuda DLLs. Feel free to change the libraries if using it in other operative systems.
- In order to install this application you'll need to have [Anaconda](https://www.anaconda.com/download) installed.

### Installation

1. Make sure you have installed the latest cuda drivers
2. Clone repo
```
git clone https://github.com/DiegoCabreraN/super-resolution
```
3. Create conda environment with the dependencies
```
conda env create -f environment_config.yml
```
4. Activate environment
```
conda env activate super-resolution
```
5. Start streamlit app
```
streamlit run app.py
```