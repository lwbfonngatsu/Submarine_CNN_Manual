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
    ```
    $ pip install tensorflow
    ```
    with Anaconda
    1. Create a conda environment named tensorflow.
    ```
    $ conda create -n tensorflow pip python=2.7 # or python=3.3, etc.
    ```
    2. Activate the conda environment.
    ```
    $ source activate tensorflow
    ```
    3. Issue a command of the following format to install TensorFlow inside your conda environment.
    ```
    (targetDirectory)$ pip install --ignore-installed --upgrade TF_PYTHON_URL
    ```
    where TF_PYTHON_URL is the [URL of the TensorFlow Python package.](https://www.tensorflow.org/install/install_mac#the_url_of_the_tensorflow_python_package)
  - Keras
    ```
    $ sudo pip install keras
    ```
  - numpy
    ```
    $ sudo pip install numpy
    ```
  - opencv
    ```
    $ sudo pip install opencv-contrib-python
    ```
3. Installation for Ubuntu
  - Tensorflow GPU
    with pip
    ```
    $ sudo pip install -U tensorflow   # Python 2.7
    $ sudo pip3 install -U tensorflow  # Python 3.n
    ```
    with Anaconda
    1. Create a conda environment named tensorflow to run a version of Python.
    ```
    $ conda create -n tensorflow pip python=2.7 # or python=3.3, etc.
    ```
    2. Activate the conda environment.
    ```
    $ source activate tensorflow
    ```
    3. Issue a command of the following format to install TensorFlow inside your conda environment.
    ```
    (tensorflow)$ pip install --ignore-installed --upgrade tfBinaryURL
    ```
    where tfBinaryURL is the [URL of the TensorFlow Python package](https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package).
  - Keras
    ```
    $ sudo pip install keras
    ```
  - numpy
    ```
    $ sudo pip install numpy
    ```
  - opencv
    ```
    $ sudo pip install opencv-contrib-python
    ```




### Prepare your dataset.
Dataset: [Download](https://mega.nz/#!5p0HhQqb!igfsGwizA2ePu1pDHxXnfK3WFpgpyuU7rXXAU1GGswk)

Inside your train, test and valid directories should be like.

![alt text](https://github.com/lwbfonngatsu/Submarine_CNN_Manual/blob/master/train_inside.png)
Each directories is contain datasets for each classes.
For me its contain [nmo, nm1, nm2, nm3, nm4, nm5, nm6, nm7, nm8, nm9, sp17, sp18, sp19] directories,
it's mean i have 13 classes name nmo, nm1, nm2, nm3, nm4, nm5, nm6, nm7, nm8, nm9, sp17, sp18, sp19.

And inside each of your class directory will contain its class datasets.

![alt text](https://github.com/lwbfonngatsu/Submarine_CNN_Manual/blob/master/class_inside.png)
