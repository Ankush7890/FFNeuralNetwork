# Feed Forward Neural Network from scratch

This project has been implemented using two ways.

- withEigen: Using Eigen library [ Eigen ]
- withoutEigen: This part is based on vector library in std namespace [ std::vector ]

[ Eigen ]: https://eigen.tuxfamily.org.
[ std::vector ]: https://en.cppreference.com/w/cpp/container/vector.

# Available activation function choices.

1. SoftMax
2. ReLU 
3. Tanh
4. Sigmoid
5. Linear 

# Available loss function choices.

1. CrossEntropy
2. MeanSquare 
3. FocalLoss
4. FocalLoss_b 

# Available optimizer choices.

1. Standard gradient descent.
2. Standard gradient descent with momentum.

Please look at the documentation for further details. 

# Installation.

````sh
    mkdir build
    cd build
    cmake -DINSTALL_PREFIX=/path/to/install/FFNeuralNetwork/ /path/to/FFNeuralNetwork/source
    make
    make install (optional)
````
In case, build is required for the version with the eigen integeration. Add the following flag.
````sh
    mkdir build
    cd build
    cmake -DWITHOUTEIGEN=OFF . ..
    make
````
If you also want to compile and run the tests, also add this -DTESTS=ON. This option needs internet access, because the process will download the googletest library.

# Getting started.

Please refer to some of examples provided along this package. These examples illustrate a way on how to create an architect for the neural network, and train it on a dataset. You might need to download the [iris dataset]( https://archive.ics.uci.edu/ml/datasets/iris ) from University of California Irvine dataset repository to see these examples play. Please refer to the documentation for further information.

# Build the documentation.

- Install doxygen 
- To build the documentation.

````sh
    mkdir build
    cd build
    cmake ..
    make docs
````
