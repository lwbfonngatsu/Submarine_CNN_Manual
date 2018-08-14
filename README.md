# Submarine_CNN_Manual
Submarine CNN installation and manual.
## Train a CNN Model.
1. Installation
    ###### Windows
  - Tensorflow GPU with [CUDA](https://developer.nvidia.com/cuda-downloads).
    
    with pip
    ```
    pip install tensorflow-gpu
    ```
    with Anaconda
    1. Create a conda environment named tensorflow (change python to your python version).
    ```
    C:> conda create -n tensorflow pip python=3.5 
    ```
    2. Activate the conda environment.
    ```
    C:> activate tensorflow
    ```
    3. Issue the appropriate command to install TensorFlow inside your conda environment.
    ```
    (tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu 
    ```
  - keras
    ```
    pip install keras
    ```
  - numpy
    ```
    pip install numpy
    ```
  - opencv
    ```
    pip install opencv-contrib-python
    ```
2. Example code for train CNN model.
  ```
  from keras.models import Sequential
  from keras.layers import Dense, Flatten
  from keras.preprocessing.image import ImageDataGenerator
  from keras.optimizers import Adam
  from keras.applications import vgg16
  from keras.callbacks import ModelCheckpoint
  
  model = Sequential()
  
  
  ```
   
