# Submarine CNN Manual
Submarine CNN installation and manual.
## Train a CNN Model.
1. Installation for Windows
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
2. Installation for MacOS
  - Tensorflow CPU
    with pip
    '''
    $ pip install tensorflow
    '''
    with Anaconda
    1. Create a conda environment named tensorflow.
    '''
    $ conda create -n tensorflow pip python=2.7 # or python=3.3, etc.
    '''
    2. Activate the conda environment.
    '''
    $ source activate tensorflow
    '''
    3. Issue a command of the following format to install TensorFlow inside your conda environment.
    '''
    (targetDirectory)$ pip install --ignore-installed --upgrade TF_PYTHON_URL
    '''
    where TF_PYTHON_URL is the [URL of the TensorFlow Python package.](https://www.tensorflow.org/install/install_mac#the_url_of_the_tensorflow_python_package)
    
3. Installation for Ubuntu

    
