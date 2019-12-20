 
#include <string>
#include <helper_functions.h>
#include <matrixOperations.h>
#include <optimizers.h>
#include <memory>

using std::vector;
using std::string;

typedef vector < double > vecD_;
typedef vector < vector < double > > vecVecD_;
typedef vector < vector < double* > > vecVecDpnt_;

/**
 * @file
 *
 * @class CreateNeuralNet
 *
 * @brief Class for creating a simple feedforward neural network, with the flexibility of choice of number of layers,
 * nodes, activation function on each layer, and kind of optimizer used for training. 
 *
 * @details This class allows to create a Neural network. The constructor needs the information of dimension of the input
 * vector, output vector and examples used in each mini batch. Then one can add further layers using addLayer method. The 
 * dimensions of last layer added should match with out_dims parameter. 
 *
 * @param input_dims dimensions of the input vectors.
 * @param out_dims dimensions of output vectors.
 * @param egPerBatch examples per batch.
 * 
 **/

class CreateNeuralNet{

private:
    //! vector storing the list of activation functions.    
    vector < void (*)( vecVecD_&, vecVecD_& ) > activation_FuncList;
    vector < void (*)( vecVecD_&, vecVecD_&, vecVecD_& ) > derivative_FuncList;
    
    //!stores the input values
    vector < vecVecD_ > inputData;
    
    //!stores all the y values
    vector < vecVecD_ > y_values;
        
    float learning_rate;
    
    vector < vecVecD_ > weights;
    vecVecD_ bias;
    
    //!stores the output of each layers.    
    vector <vecVecD_> layerOuts;
    
    //!pointer to the loss function.
    double (*loss_Function)(vecVecD_&,vecVecD_&);
    vecVecD_ (*der_lossFunction)(vecVecD_&,vecVecD_&);
        
    //!pointer to the optimizer object.
    std::shared_ptr <BaseOptimizer> optimizer_obj;
    
    //!dW
    vector < vecVecD_ > deltaWeights;
    
    //!dZ (don't need d(bias)) as dZ already has that information.
    vector <vecVecD_ > deltaZ;
    
    //!some useful things.
    int32_t lastLayerDims,lastLayerIndex;

public:
    
    //!variables for Neural Network 
    int32_t input_dims, egPerBatch, out_dims, miniBatch;
    
    //!constructor
    CreateNeuralNet(int32_t input_dims=2, int32_t out_dims=2, int32_t egPerBatch=1);
    
    //!Function for adding layers with specification to add some function
    void addLayer(int32_t nodes=1, string activationFunction="ReLU");
    
    void forwardProp();
    void backwardProp();
    void optimize();
    
    //!train accuracy finder
    double trainAcc();
    
    //!train loss finder
    double calculateLoss();

    //!Initialize the weights randomly
    void randomWeightInit();
    
    //!File reader
    void readFile(string filename, int32_t totEgs);
    
    //!Fit function
    void fit(int32_t epochs = 100, float learning_rate=0.1, string lossFunction_="cross_entropy", string optimizer="GradientDescent", float alpha=0.0, float alpha_momentum=0.0);
};

CreateNeuralNet::CreateNeuralNet(int32_t input_dims_,int32_t out_dims_, int32_t egPerBatch_)
:input_dims(input_dims_),
out_dims(out_dims_),
egPerBatch(egPerBatch_)
{
    //Initialize the dimension of weight, bias, and gradients that stores the layer number to 0 
    weights.resize(0);
    bias.resize(0);

    deltaWeights.resize(0);
    deltaZ.resize(0);
    
    
    //See in addLayer function
    lastLayerDims=input_dims;
}

/**
* @brief Adds a layer to the neural network.
* 
* @param nodes number of nodes required in the layer added.
* @param activationFunction string value of activation function required for layer. See helper_functions.h for available 
* choice of activation functions.
*/

void CreateNeuralNet::addLayer(int32_t nodes, string activationFunction){
    
    //create two dummy placeholders
    vecD_ oneD_dummy_Vec;
    vecVecD_ twoD_dummy_Vec;
    
    oneD_dummy_Vec.resize(nodes,0);
   
    //placeholders for 1D vectors for every layer
    bias.push_back(oneD_dummy_Vec);
    
    //placeholders for 2D vectors for every layer
    twoD_dummy_Vec.resize(lastLayerDims, vecD_(nodes, 0));
    
    weights.push_back(twoD_dummy_Vec);
    deltaWeights.push_back(twoD_dummy_Vec);
   
    //place holders for layerOuts
    twoD_dummy_Vec.resize(0);
    twoD_dummy_Vec.resize(egPerBatch, vecD_(nodes, 0));
    layerOuts.push_back(twoD_dummy_Vec);
    deltaZ.push_back(twoD_dummy_Vec);
    
    lastLayerDims=nodes;
    
    //activation functions for the layers
    if(activationFunction=="ReLU"){
        
        void (*functionPointer)(vecVecD_&, vecVecD_&) = ReLU;
        activation_FuncList.push_back(functionPointer);
        void (*functionPointer2)(vecVecD_&, vecVecD_&, vecVecD_&) = dReLU;
        derivative_FuncList.push_back(functionPointer2);}
        
    else if(activationFunction=="softMax"){
        
        void (*functionPointer)(vecVecD_&,vecVecD_&) = softMax;
        activation_FuncList.push_back(functionPointer);
        void (*functionPointer2)(vecVecD_&, vecVecD_&, vecVecD_&) = dSoftMax;
        derivative_FuncList.push_back(functionPointer2);}
        
    else if(activationFunction=="sigmoid"){
        
        void (*functionPointer)(vecVecD_&,vecVecD_&) = sigmoid;
        activation_FuncList.push_back(functionPointer);
        void (*functionPointer2)(vecVecD_&, vecVecD_&, vecVecD_&) = dSigmoid;
        derivative_FuncList.push_back(functionPointer2);}

    else if(activationFunction=="tanh"){
        
        void (*functionPointer)(vecVecD_&,vecVecD_&) = tanh;
        activation_FuncList.push_back(functionPointer);
        void (*functionPointer2)(vecVecD_&, vecVecD_&, vecVecD_&) = dTanh;
        derivative_FuncList.push_back(functionPointer2);}
        
    else if(activationFunction=="linear"){
        
        void (*functionPointer)(vecVecD_&,vecVecD_&) = linear;
        activation_FuncList.push_back(functionPointer);
        void (*functionPointer2)(vecVecD_&, vecVecD_&, vecVecD_&) = dLinear;
        derivative_FuncList.push_back(functionPointer2);}
        
    else{
        std::cout<<activationFunction<<"\n";
        throw std::runtime_error("Wrong entry for activation function");}
}

/**
* @brief Performs the training on the chosen architecture of neural network.
* 
* @param epochs number of epochs to train on.
* @param learning_rate_ learning rate during the training.
* @param lossFunction_ string value choice of the loss function, see helper_functions.h 
* for choices of the loss function.
* @param optimizer string value for the choice optimizer.
* @param alpha momentum in range [0,1] for Gradient Descent With Momentum optimizer. 
* @param alpha_momentum momentum in range [0,1] for Gradient Descent With Momentum optimizer. 
*/


void CreateNeuralNet::fit(int32_t epochs, float learning_rate_, string lossFunction_, string optimizer, float alpha, float alpha_momentum){
    
    //learning rate can be dynamically changed with schedulers.
    learning_rate=learning_rate_;
    
    lastLayerIndex=weights.size()-1;
    
    //The loss function
    if(lossFunction_=="meanSquared"){
        loss_Function=meanSquare;
        der_lossFunction=DmeanSquare;
    }
    else if(lossFunction_=="cross_entropy"){
        loss_Function=crossEntropy;
        der_lossFunction=DcrossEntropy;
    }
    else{
        std::cout<<loss_Function<<"\n";
        throw std::runtime_error("Wrong entry for loss_Function");}
    
    //The optimizers
   if(optimizer=="GradientDescent")
       optimizer_obj = std::make_shared <GradientDescent>(learning_rate, &weights, &bias, &deltaWeights, &deltaZ);
   
   else if(optimizer=="GradientDescentWithMomentum"){
       optimizer_obj = std::make_shared <GradientDescentWithMomentum>(learning_rate, &weights, &bias, &deltaWeights, &deltaZ);
       optimizer_obj->set_alpha(alpha_momentum);
   }
   else{
       std::cout<<optimizer<<"\n";
       throw std::runtime_error("Wrong entry for optimizer");
   }
    
    //randomly initiate weights
    randomWeightInit();

    for(int32_t i=0;i<epochs;i++){
        
        //go over the miniBatches.
        for(miniBatch=0; miniBatch<inputData.size(); miniBatch++){
            forwardProp();
            backwardProp();
            optimizer_obj->optimize();
        }
        
        if(i%1000==0){
            std::cout<<loss_Function(y_values[0],layerOuts[lastLayerIndex])<<" "<<i<<" <-loss\n";}
    }
}


/**
* @brief Calculates the loss using assigned loss function.
*/

double CreateNeuralNet::calculateLoss(){
        
    return loss_Function(y_values[0],layerOuts[lastLayerIndex]);
}    


/**
* @brief Calculates the Training Accuracy for classification problems.
*/

double CreateNeuralNet::trainAcc(){
    
    double acc=0;
    for(miniBatch=0; miniBatch<inputData.size(); miniBatch++){
        forwardProp();        
        for(int32_t i=0;i<y_values[miniBatch].size();i++)
            acc += fabs(floor(layerOuts[lastLayerIndex][i][0]+0.5)-y_values[miniBatch][i][0]);
    }
    
    return 1-acc/(egPerBatch*y_values.size());
}    

/**
* @brief For randomly initiating weights.
*/

void CreateNeuralNet::randomWeightInit(){
    
    for(int32_t layerNum=0; layerNum<weights.size(); layerNum++)
        for(int32_t dim1=0; dim1<weights[layerNum].size(); dim1++)
            for(int32_t dim2=0; dim2<weights[layerNum][dim1].size(); dim2++)
                weights[layerNum][dim1][dim2] = 0.01*((double) rand() / (RAND_MAX) -0.5);
}    

/**
* @brief Read the file in the csv format.
* @param filename filename to read string value.
* @param totEgs total number of example to be read.
*/

void CreateNeuralNet::readFile(string filename, int32_t totEgs){
    
    std::ifstream fileReader(filename);
    std::string numstring;
    
    for(int32_t egNum=0; egNum<totEgs; egNum = egNum+egPerBatch){

        vecVecD_ twoD_dummy_Vec;
        twoD_dummy_Vec.resize(egPerBatch, vecD_(input_dims));
        inputData.push_back(twoD_dummy_Vec);
       
        vecVecD_ twoD_dummy_Vec2;
        twoD_dummy_Vec2.resize(egPerBatch, vecD_(out_dims));
        y_values.push_back(twoD_dummy_Vec2);

        //store for input data
        for(int32_t egBatch=0;egBatch<egPerBatch;egBatch++){
            
            for(int32_t featureNum=0; featureNum < input_dims; featureNum++){
                getline(fileReader, numstring, ',');
                inputData[int(egNum/egPerBatch)][egBatch][featureNum]=(std::stod(numstring));
            }
            
            //store for output data
            getline(fileReader, numstring,'\n');
            
            int32_t value=std::stoi(numstring);
            
            //one hot encoding
            for(int32_t i=0;i<out_dims;i++){
                if(i==value)
                    y_values[egNum/egPerBatch][egBatch][i]=1;
                else
                    y_values[egNum/egPerBatch][egBatch][i]=0;
            }
        }
        
    }
    
}

/**
* @brief Forward propagator.
*/

void CreateNeuralNet::forwardProp(){
    
    //X*W
    vecVecD_ preFactor = matMul(inputData[miniBatch], weights[0]);  
    
    //X*W+bias
    matAdd(preFactor, bias[0], layerOuts[0]);
    
    //a=G(X*W+bias) activation function
    activation_FuncList[0](layerOuts[0], layerOuts[0]);

    for(int32_t layerNum = 1; layerNum<layerOuts.size(); layerNum++){
        
        //a_lastLayer*W_thislayer
        vecVecD_ preFactor = matMul(layerOuts[layerNum-1], weights[layerNum]);
        
        //a_lastLayer*W_thislayer+bias
        matAdd(preFactor, bias[layerNum], layerOuts[layerNum]);
        
        //a_thislayer=G(a_lastLayer*W_thislayer+bias) activation function
        activation_FuncList[layerNum](layerOuts[layerNum], layerOuts[layerNum]);
    }
}

/**
* @brief Back propagator.
*/


void CreateNeuralNet::backwardProp(){
        
    //derivative of loss function
    vecVecD_ d_loss = der_lossFunction(y_values[miniBatch], layerOuts[lastLayerIndex]);

    //derivative of activation function of last layer
    derivative_FuncList[lastLayerIndex](d_loss,layerOuts[lastLayerIndex], deltaZ[lastLayerIndex]);

    //transpose of last layer outputs
    vecVecD_ layerTrans = matTrans(layerOuts[lastLayerIndex-1]);
    
    //finally d(W)=a_Trans*dZ
    matMul(layerTrans, deltaZ[lastLayerIndex], deltaWeights[lastLayerIndex]);
    
    for(int32_t layerNum=lastLayerIndex-1; layerNum>=0; layerNum--){

        //transpose of weights from the forward layer
        vecVecD_ layerTransForw = matTrans(weights[layerNum+1]);

        //multiplying preFactor to multiplied with da/dz
        vecVecD_ preFactor = matMul(deltaZ[layerNum+1],layerTransForw);


        // a'(z)=da/dz as function of a
        derivative_FuncList[layerNum](preFactor, layerOuts[layerNum], deltaZ[layerNum]);

        
        //transpose of layer from the back layer
        vecVecD_ layerTransBack;
        
        //if layer number=0, then input layers is the last output layer
        if(layerNum)
            layerTransBack = matTrans(layerOuts[layerNum-1]);
        else
            layerTransBack = matTrans(inputData[miniBatch]);
        //finally d(W)
        matMul(layerTransBack, deltaZ[layerNum], deltaWeights[layerNum]);
    }
    
}
