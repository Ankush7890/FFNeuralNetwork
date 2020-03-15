# Feed Forward Neural Network from scratch

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
In case you only want the version with the with the eigen integeration. Add the following flag.
````sh
    mkdir build
    cd build
    cmake -DWITHOUTEIGEN=OFF . ..
    make
````

