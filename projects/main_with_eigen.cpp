#include <NNConfig.h>
#include <vector>
#include <iostream>

#include <stdlib.h>
#include "cmath"
#include "stdint.h"
#include <fstream>
#include <string>
#ifdef WITHOUTEIGEN
#include <withoutEigen/CreateNeuralNet.h>
#else
#include <withEigen/CreateNeuralNet.h>
#endif
#include <chrono> 

using std::string;
using namespace std::chrono;

int main(void){
        
    int32_t input_dims=2;
    int32_t egsPerBatch=100;
    
    CreateNeuralNet NN1(input_dims, 2, egsPerBatch);
    
    int32_t nodes=6;
    string activationFunction="Tanh";
    
    NN1.addLayer(nodes, activationFunction);

    nodes=2;
    activationFunction="SoftMax";
    
    NN1.addLayer(nodes, activationFunction);
    
    NN1.readFile("iris.csv",100);
    
    auto start = high_resolution_clock::now(); 
    
    NN1.fit(10000, 0.004, "CrossEntropy");
    
    auto stop = high_resolution_clock::now(); 
     
    auto duration = duration_cast<microseconds>(stop - start); 
  
    std::cout << "Time taken by function: "
         << duration.count() << " microseconds\n"; 
    std::cout<<NN1.trainAcc()<<"\n";
}

