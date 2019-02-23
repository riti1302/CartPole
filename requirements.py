#The gym library a collection of test problems — environments — that you can use to work
# out your reinforcement learning algorithms. These environments have a shared interface,
#allowing you to write general algorithms.
gym

#### ESSENTIAL LIBRARIES FOR MAIN FUNCTIONALITY ####

# Neural net and related library.
## Tensorflow Requirements ##
# Tensorflow is required to run this code but depends on specific configurations. Install from:
# https://www.tensorflow.org/get_started/os_setup#overview
# If you want to use the GPU version, you will also need Nvidia's CUDA toolkit and cuDNN:
# https://developer.nvidia.com/cuda-downloads
# https://developer.nvidia.com/cudnn
# Note that if you want to use the GPU version, you have to `pip uninstall tensorflow`
# and `pip install tensorflow-gpu`, since both cannot coexist. 
tensorflow>=1.0.1

# For data manipulation
numpy
