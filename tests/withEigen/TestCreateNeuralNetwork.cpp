
/*****************************************************************************/
/**
 * @file
 * @brief Tests for the class withEigen/CreateNeuralNet
 *
 * */
/*****************************************************************************/

#include "gtest/gtest.h"

#include <NNConfig.h>
#include <vector>
#include <iostream>

#include <stdlib.h>
#include "cmath"
#include "stdint.h"
#include <fstream>
#include <string>
#include <withEigen/CreateNeuralNet.h>
#include <chrono> 


using namespace std;

/************************************************************************/
//define test fixtures for the different tests their purpose is to set up
//the tests to suppress cout's output such that is does not display on the
//standard output during the tests. this makes google test's output more readeable
/************************************************************************/

class TestCreateNeuralNetwork: public ::testing::Test{
public:

  //redirect cout output
  virtual void SetUp(){
    originalBuffer=cout.rdbuf();
    cout.rdbuf(tempStream.rdbuf());
  };

  //restore original output
  virtual void TearDown(){
    cout.rdbuf(originalBuffer);
  };
  
private:
  std::streambuf* originalBuffer;
  std::ostringstream tempStream;

};

int32_t input_dims=2;
int32_t egsPerBatch=100;
int32_t nodes;

//end to end test using specific activation functions and loss functions.

TEST(TestCreateNeuralNetwork, CheckWithTanhAndSoftMaxAndCrossEntropy)
{
    
    CreateNeuralNet NN(input_dims, 2, egsPerBatch);
    
    int32_t nodes=6;
    string activationFunction="Tanh";
    
    NN.addLayer(nodes, activationFunction);

    nodes=2;
    activationFunction="SoftMax";
    
    NN.addLayer(nodes, activationFunction);

    
    NN.readFile("tests/iris.csv",100);
    //TODO cleaner version to initiate variables
    NN.fit(0, 0.004, "CrossEntropy");

    float loss_init=NN.trainLoss();
    
    NN.fit(10000, 0.004, "CrossEntropy");
    
    ASSERT_LT(NN.trainLoss(), loss_init);

 }

TEST(TestCreateNeuralNetwork, CheckWithReLUAndMeanSquare)
{
    
    CreateNeuralNet NN(input_dims, 1, egsPerBatch);
    
    int32_t nodes=6;
    string activationFunction="ReLU";
    
    NN.addLayer(nodes, activationFunction);

    nodes=1;
    activationFunction="ReLU";
    
    NN.addLayer(nodes, activationFunction);

    
    NN.readFile("tests/iris.csv",100,0,0);
    //TODO cleaner version to initiate variables
    NN.fit(0, 0.004, "MeanSquare");

    float loss_init=NN.trainLoss();
    
    NN.fit(10000, 0.004, "MeanSquare");
    
    ASSERT_LT(NN.trainLoss(), loss_init);

 }

TEST(TestCreateNeuralNetwork, CheckWithSquareAndFocalLoss)
{
    
    CreateNeuralNet NN(input_dims, 2, egsPerBatch);
    
    int32_t nodes=6;
    string activationFunction="Sigmoid";
    
    NN.addLayer(nodes, activationFunction);

    nodes=2;
    activationFunction="SoftMax";
    
    NN.addLayer(nodes, activationFunction);

    
    NN.readFile("tests/iris.csv",100);
    //TODO cleaner version to initiate variables
    NN.fit(0, 0.004, "FocalLoss");

    float loss_init=NN.trainLoss();
    
    NN.fit(10000, 0.004, "FocalLoss");
    
    ASSERT_LT(NN.trainLoss(), loss_init);

 }

 TEST(TestCreateNeuralNetwork, CheckWithLinearAndReLU)
{
    
    CreateNeuralNet NN(input_dims, 1, egsPerBatch);
    
    int32_t nodes=6;
    string activationFunction="Linear";
    
    NN.addLayer(nodes, activationFunction);

    nodes=1;
    activationFunction="ReLU";
    
    NN.addLayer(nodes, activationFunction);

    
    NN.readFile("tests/iris.csv",100);
    //TODO cleaner version to initiate variables
    NN.fit(0, 0.004, "MeanSquare");

    float loss_init=NN.trainLoss();
    
    NN.fit(10000, 0.004, "MeanSquare");
    
    ASSERT_LT(NN.trainLoss(), loss_init);

 }
