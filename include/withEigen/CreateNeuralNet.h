#include <string>
#include <withEigen/helper_functions.h>
#include <Eigen/Dense>
#include <withEigen/optimizers.h>
#include <fstream>
using std::vector;
using std::string;
using namespace Eigen;

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::ArrayXXd;

typedef vector < MatrixXd > vecM_;
typedef vector < RowVectorXd > vecR_;

typedef vector < ArrayXXd > vecA_;
typedef vector < Array<double, 1, Dynamic>  > vecAR_;

/**
 *
 * @class String_2_Class
 *
 * @brief Class for creating a map between the string value and the object that belongs that class. Only useful for brevity purposes.
 * Note: shared pointer only useful as all the activation functions are simply collection of methods. In case activation function have
 * data structure associated with them. This map might create undefined behavior.
 * 
 **/

struct String_2_Class{
    
    String_2_Class(){
        mp["Sigmoid"] = std::make_shared <Sigmoid>();
        mp["Tanh"] = std::make_shared <Tanh>();
        mp["ReLU"] = std::make_shared <ReLU>();
        mp["Linear"] = std::make_shared <Linear>();
        mp["SoftMax"] = std::make_shared <SoftMax>();
        
        mp["CrossEntropy"] = std::make_shared <CrossEntropy>();
        mp["MeanSquare"] = std::make_shared <MeanSquare>();
        mp["FocalLoss"] = std::make_shared <FocalLoss>();
        mp["FocalLoss_b"] = std::make_shared <FocalLoss_b>();
    }
    
    map <string, std::shared_ptr<BaseFn> > mp;
};

//!Generate the map.
String_2_Class string_2_class;


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
    vector < std::shared_ptr<BaseFn> > activation_FuncList;
    
    //!stores all the input values
    vecA_ inputData;

    //!stores all the input values for validation
    vecA_ valData;
    
    //!stores all the y values for validation
    vecA_ val_y_values;
        
    //!stores all the y values
    vecA_ y_values;
        
    //
    float learning_rate;
    
    vecA_ weights;
    
    vecAR_ bias;
    
    //!stores the out of each layers.    
    vecA_ layerOuts;
        
    //!dW.
    vecA_ deltaWeights;
    
    //!loss function. 
    std::shared_ptr<BaseFn> loss_Function;
    
    //!optimizer function. 
    std::shared_ptr<BaseOptimizer> optimizer_obj;
    
    std::stringstream filenameWeights,filenameAcc;
    std::ofstream writeFileAcc;

    //!dZ (don't need d(bias) as dZ already has information of db.
    vecA_ deltaZ;
    
    //!whether to one hot encode or not.
    bool OHE;
    
    //!some useful things.
    int32_t lastLayerDims,lastLayerIndex,miniBatchNum;

public:
    
    //!init values 
    int32_t input_dims,egPerBatch,output_dims;
    
    CreateNeuralNet(int32_t input_dims=2, int32_t output_dims=2, int32_t egPerBatch=1, double epsilon_=0.1);
    
    //Function for adding layers with specification to add some function
    void addLayer(int32_t nodes=1, string activationFunction="ReLU");
    
    //obvious things
    void forwardProp(vecA_ &datType);
    void backwardProp();
    void optimize();
    
    //findin the train accuracy
    double trainAcc();


    //findin the training loss
    double trainLoss();

    //findin the val accuracy
    double valAcc();
    
    void randomWeightInit();
    
    void writeWeightsandAccuracy(int32_t epoch);
    
    //read the file
    void readFile(string filename, int32_t trainEgs, int32_t valEgs=0, int32_t OHE=1);
        
    //To perform gradient decent using forward and backward prop
    void fit(int32_t tot_epochs = 100, float learning_rate=0.1, string lossFunction_="CrossEntropy", int32_t save_interval=1000, string optimizer="GradientDescentWithMomentum", double gamma_focal=0, double alpha_focal=0, double alpha_momentum=0);
};

//constructor
CreateNeuralNet::CreateNeuralNet(int32_t input_dims_,int32_t output_dims_, int32_t egPerBatch_, double epsilon_)
:input_dims(input_dims_),
output_dims(output_dims_),
egPerBatch(egPerBatch_)
{
    
    //Initialize the dimension of weight, bias, and gradients 
    //that stores the layer number to 0 
    weights.resize(0);
    bias.resize(0);

    deltaWeights.resize(0);
    deltaZ.resize(0);
        
    //this would help later in addLayer function
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
        
    Array<double, 1, Dynamic> oneD_dummy_Vec(nodes);
    
    //placeholders for variables which only 
    //need 1D vectors for every layer
    bias.push_back(oneD_dummy_Vec);
    
    //placeholders for variables which 
    //need 2D vectors for every layer
    ArrayXXd twoD_dummy_Vec = ArrayXXd::Random(lastLayerDims, nodes);   
    weights.push_back(twoD_dummy_Vec);
    deltaWeights.push_back(twoD_dummy_Vec);
   
    //for layerOuts place holders are of different size
    ArrayXXd twoD_dummy_Vec2(egPerBatch, nodes);
    
    
    layerOuts.push_back(twoD_dummy_Vec2);
    deltaZ.push_back(twoD_dummy_Vec2);
    
    lastLayerDims=nodes;

    activation_FuncList.push_back(string_2_class.mp.at(activationFunction));
}

/**
* @brief For randomly initiating weights.
*/

void CreateNeuralNet::randomWeightInit(){
    
    for(int32_t layerNum=0; layerNum<weights.size(); layerNum++){
     
        for(int32_t dim1=0; dim1<weights[layerNum].rows(); dim1++)
            for(int32_t dim2=0; dim2<weights[layerNum].cols(); dim2++)
                weights[layerNum](dim1,dim2) = 0.1*((double) rand() / (RAND_MAX) -0.5);
        
        for(int32_t dim1=0; dim1<bias[layerNum].rows(); dim1++)
            for(int32_t dim2=0; dim2<bias[layerNum].cols(); dim2++)
                bias[layerNum](dim1,dim2) = 0;
    }       
}    

/**
* @brief Performs the training on the chosen architecture of neural network.
* 
* @param tot_epochs total number of epochs to train on.
* @param learning_rate_ learning rate during the training.
* @param lossFunction_ string value choice of the loss function, see helper_functions.h 
* for choices of the loss function.
* @param optimizer string value for the choice optimizer.
* @param alpha_focal alpha value in the focal loss function.
* @param gamma_focal gamma value in the focal loss function.
* @param alpha_momentum momentum in range [0,1] for Gradient Descent With Momentum optimizer. 
*/

void CreateNeuralNet::fit(int32_t tot_epochs, float learning_rate_, string lossFunction_, int32_t save_interval, string optimizer, 
                          double gamma_focal, double alpha_focal, double alpha_momentum){
    
    //learning rate can be dynamically changed with schedulers.
    learning_rate=learning_rate_;
    
    lastLayerIndex = weights.size()-1;
    
    loss_Function = string_2_class.mp.at(lossFunction_);

   if(lossFunction_=="FocalLoss" or lossFunction_=="FocalLoss_b")
        loss_Function->set_alpha_gamma(alpha_focal, gamma_focal);

   if(optimizer=="GradientDescent")
       optimizer_obj = std::make_shared <GradientDescent>(learning_rate, &weights, &bias, &deltaWeights, &deltaZ);
   
   else if(optimizer=="GradientDescentWithMomentum"){
       optimizer_obj = std::make_shared <GradientDescentWithMomentum>(learning_rate, &weights, &bias, &deltaWeights, &deltaZ);
       optimizer_obj->set_alpha(alpha_momentum);
   }
   //start the file for saving the accuracy
   filenameAcc<<"Acc_inD_"<<input_dims<<"_layers_"<<lastLayerIndex+1<<"_egPerBatch_"<<egPerBatch<<".dat";
   
   writeFileAcc.open(filenameAcc.str().c_str());

   for(int32_t epoch=0; epoch<tot_epochs; epoch++){
        
        //go over in mini batches.
        for(miniBatchNum=0;miniBatchNum<inputData.size();miniBatchNum++){  
            forwardProp(inputData);
            backwardProp();
            optimizer_obj->optimize();
        }
        
        if(epoch%save_interval==0)
            writeWeightsandAccuracy(epoch);
    }
    
}

/**
* @brief Saves the weights and accuracy.
* 
* @param epochs epochs associated with the weight. It will be used in naming of the file.
* 
*/
void CreateNeuralNet::writeWeightsandAccuracy(int32_t epoch){
    
    filenameWeights.str("");
    filenameWeights<<"Weight_inD_"<<input_dims<<"_layers_"<<lastLayerIndex+1<<"_iter_"<<epoch<<".dat";
    
    std::ofstream writeFileWeights;
    writeFileWeights.open(filenameWeights.str().c_str());
    
    double accLoss=0,valLoss=0;
    for(miniBatchNum=0; miniBatchNum<inputData.size(); miniBatchNum++){
        forwardProp(inputData);
        accLoss += loss_Function->loss_f(y_values[miniBatchNum], layerOuts[lastLayerIndex]);
        std::cout<<epoch<<"  "<<accLoss<<"\n";

    }

    for(miniBatchNum=0; miniBatchNum<valData.size(); miniBatchNum++){
        forwardProp(valData);
        valLoss += loss_Function->loss_f(val_y_values[miniBatchNum],layerOuts[lastLayerIndex]);
    }

    writeFileAcc<<valLoss/valData.size()<<"  "<<accLoss/inputData.size()<<"   "<<valAcc()<<"  "<<trainAcc()<<"\n";
    writeFileAcc.flush();
    
    for(int32_t lay=0; lay<weights.size(); lay++)
        for(int32_t rs=0; rs<weights[lay].rows(); rs++){
            
            for(int32_t cls=0; cls<weights[lay].cols()-1; cls++)
                writeFileWeights<<weights[lay](rs,cls)<<",";
            
            writeFileWeights<<weights[lay](rs, weights[lay].cols()-1)<<"\n";}
            
}


/**
* @brief Calculates the Training Accuracy for classification problems.
*/

double CreateNeuralNet::trainLoss(){

    double trainLoss=0;
    for(miniBatchNum=0; miniBatchNum<inputData.size(); miniBatchNum++){
        forwardProp(inputData);
        trainLoss += loss_Function->loss_f(y_values[miniBatchNum], layerOuts[lastLayerIndex]);

    }
    
    return trainLoss/inputData.size();
}


/**
* @brief Calculates the Training Accuracy for classification problems.
*/

double CreateNeuralNet::trainAcc(){

    double acc=0;
    
    for(miniBatchNum=0; miniBatchNum<inputData.size();miniBatchNum++){
        forwardProp(inputData);
        acc += abs(floor(layerOuts[lastLayerIndex]+0.5) - y_values[miniBatchNum]).sum();
        }
    
    return 1-acc/(output_dims*inputData.size()*egPerBatch);
}

/**
* @brief Calculates the Validation Accuracy for classification problems.
*/

double CreateNeuralNet::valAcc(){
    double acc=0;
    
    for(miniBatchNum=0; miniBatchNum<valData.size();miniBatchNum++){
        forwardProp(valData);
        acc += abs(floor(layerOuts[lastLayerIndex]+0.5) - val_y_values[miniBatchNum]).sum();
        }
    
    return 1-acc/(output_dims*inputData.size()*egPerBatch);
}    
/**
* @brief Read the file in the csv format.
* 
* @param filename filename to read string value.
* @param trainEgs total number of training examples to be read.
* @param valEgs total number of validation examples to be read.
* @param OHE Whether to one hot encode or not.
* 
* Note: The total number of examples in the file should not be larger than trainEgs+valEgs, 
* otherwise it might generate undefined behavior.
*/


void CreateNeuralNet::readFile(string filename, int32_t trainEgs, int32_t valEgs, int32_t OHE){
    
    std::ifstream fileReader(filename);
    std::string numstring;

    ArrayXXd dummy(egPerBatch, input_dims);            
    ArrayXXd dummy2(egPerBatch, output_dims);
    
    int32_t numtrain=0,numval=0;
    
    for(int32_t egNum=0; egNum<trainEgs+valEgs; egNum++){
        
        //store for input data
        for(int32_t featureNum=0; featureNum < input_dims; featureNum++){
            getline(fileReader, numstring,',');
            dummy((egNum)%egPerBatch, featureNum) = std::stod(numstring);
        }
        
        //store for output data
        getline(fileReader, numstring,'\n');
        
        if(OHE){
            int32_t value=std::stoi(numstring);
            
            //one hot encoding
            for(int32_t i=0;i<output_dims;i++){
                if(i==value)
                    dummy2((egNum)%egPerBatch, i)=1;
                else
                    dummy2((egNum)%egPerBatch, i)=0;
            }
        }
        else{
            double value=std::stod(numstring);
            dummy2((egNum)%egPerBatch, 0) = value;
        }
        
        //save when one mini batch is done
        if(egNum && (egNum+1)%egPerBatch==0){
            
            int32_t valOrtrain=rand()%2;
            
            if(egNum<trainEgs){
                inputData.push_back(dummy);
                y_values.push_back(dummy2);}
            
            else{
                valData.push_back(dummy);
                val_y_values.push_back(dummy2);}
        }
    }
}

/**
* @brief Forward propagator.
*/

void CreateNeuralNet::forwardProp(vecA_ &datType){
    
    //X*W+bias
    layerOuts[0] = (datType[miniBatchNum].matrix()*weights[0].matrix()).array().rowwise()+bias[0];

    //a=G(X*W+bias) activation function
    activation_FuncList[0]->func(layerOuts[0], layerOuts[0]);
    
    for(int32_t layerNum = 1; layerNum<layerOuts.size(); layerNum++){
                
        //a_lastLayer*W_thislayer+bias
        layerOuts[layerNum] = (layerOuts[layerNum-1].matrix()*weights[layerNum].matrix()).array().rowwise()+bias[layerNum];
        
        //a_thislayer=G(a_lastLayer*W_thislayer+bias) activation function
        activation_FuncList[layerNum]->func(layerOuts[layerNum], layerOuts[layerNum]);        
    }

}

/**
* @brief Back propagator.
*/

void CreateNeuralNet::backwardProp(){
        
    //derivative of loss function
    ArrayXXd d_loss = loss_Function->derivative(y_values[miniBatchNum], layerOuts[lastLayerIndex]);
    
    //derivative of activation function of last layer
    activation_FuncList[lastLayerIndex]->derivative(d_loss, layerOuts[lastLayerIndex], deltaZ[lastLayerIndex]);

    ArrayXXd layerTrans = layerOuts[lastLayerIndex-1].transpose();
    
    //d(W)=a_Trans*dZ
    deltaWeights[lastLayerIndex] = layerTrans.matrix()*deltaZ[lastLayerIndex].matrix();

    for(int32_t layerNum=lastLayerIndex-1; layerNum>=0; layerNum--){

        //multiplying preFactor to multiplied with da/dz
        ArrayXXd preFactor = deltaZ[layerNum+1].matrix()*weights[layerNum+1].matrix().transpose();

        //a'(z)=da/dz as function of a
        activation_FuncList[layerNum]->derivative(preFactor, layerOuts[layerNum], deltaZ[layerNum]);

        //transpose of layer from the back layer
        ArrayXXd layerTransBack;
        
        //if layer number=0, then input layers is the last output layer
        if(layerNum)
            layerTransBack = layerOuts[layerNum-1].transpose();
        else
            layerTransBack = inputData[miniBatchNum].transpose();
        
        //d(W)
        deltaWeights[layerNum] = layerTransBack.matrix()*deltaZ[layerNum].matrix();
    }
    
}
