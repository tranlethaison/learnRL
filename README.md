# learnRL

Let's try implement Reinforcement Learning algorithms with TensorFlow 2.0.

-   [learnRL](#learnrl)
    -   [Environment](#environment)
    -   [Algorithms](#algorithms)

## Environment

Find a way to install TensorFlow 2.0 Beta on your system.  

Bellow is a simple method that works on any OS,  
without messing up existing TensorFlow environment.

-   Dependencies: [Conda](https://docs.conda.io/en/latest/miniconda.html)
-   Using GPU:
    ```shell
    $ conda create -n tf2 python cudatoolkit=10.0 cupti cudnn
    $ conda activate tf2
    $ pip install -r requirements.txt
    $ python test/test_gpu.py
    ```
-   Or not:
    ```shell
    $ conda create -n tf2 python 
    $ conda activate tf2
    $ pip install -r requirements_nogpu.txt
    ```

## RL Algorithms

-   [Simple Policy Gradient](learnrl/simple_pg.py)
    ```shell
    $ python learnrl/simple_pg.py train \
        --env_name CartPole-v0 \
        --lr 0.025 \
        --n_epochs 50 \
        --batch_size 5000 \
        --do_render 1
    ```

## Others

For the sake of learning. Let's implement Neural Networks, and training algorithms
only using Numpy.

-   [Densely connected NN](test/test_numpynn/dense_net.py)
    ```shell
    $ python test/test_numpynn/dense_net.py
    ```
