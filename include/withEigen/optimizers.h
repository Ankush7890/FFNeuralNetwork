using namespace Eigen;

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::ArrayXXd;

typedef vector < MatrixXd > vecM_;
typedef vector < RowVectorXd > vecR_;

typedef vector < ArrayXXd > vecA_;
typedef vector < Array<double, 1, Dynamic>  > vecAR_;

/**
 * @file 
 * 
 * @brief Optimizers which could used during training.
 * 
 * @details Choices:<br>
 * \b GradientDescent <br>
 * \b GradientDescentWithMomentum <br>
 * 
 **/


/**
 *
 * @class BaseOptimizer
 *
 * @brief Base class for all Optimizers.
 * 
 * @details Base functions.<br>
 * \b optimize optimizer function. <br>
 * \b set_alpha setter of the alpha value for GradientDescentWithMomentum. <br>
 * 
 **/

struct BaseOptimizer{
    
    //constructor
    BaseOptimizer(){};
    virtual void optimize(){};
    virtual void set_alpha(float alpha){};
};

struct GradientDescent : public BaseOptimizer
{

    float learning_rate, totalNumLayers;

    vecA_* weights;
    
    vecAR_* bias;
    
    vecA_* deltaZ;
    vecA_* deltaWeights;

    //constructor
    GradientDescent(float learning_rate_, vecA_* weights_, vecAR_* bias_, vecA_* deltaWeights_, vecA_* deltaZ_)
    :learning_rate(learning_rate_),
    weights(weights_),
    bias(bias_),
    deltaWeights(deltaWeights_),
    deltaZ(deltaZ_)
    {
        totalNumLayers = weights->size();        
    };
    
    void optimize(){
        
        for(int32_t layerNum=0; layerNum<totalNumLayers; layerNum++){
            
            (*weights)[layerNum] -= learning_rate*(*deltaWeights)[layerNum];
            (*bias)[layerNum] -= learning_rate*(*deltaZ)[layerNum].colwise().sum();

        }
    }
};

struct GradientDescentWithMomentum : public GradientDescent
{
   
    using GradientDescent::GradientDescent;
    using GradientDescent::learning_rate;
    using GradientDescent::totalNumLayers;
    using GradientDescent::weights;
    using GradientDescent::deltaWeights;
    using GradientDescent::deltaZ;
    using GradientDescent::bias;

    float alpha=0;
    
    vecA_ deltaZ_old;
    vecA_ deltaWeights_old;
    
    //constructor
    GradientDescentWithMomentum(float learning_rate_, vecA_* weights_, vecAR_* bias_, vecA_* deltaWeights_, vecA_* deltaZ_)
    :GradientDescent(learning_rate_, weights_, bias_, deltaWeights_, deltaZ_)
    {
        copy_all_new2old();
    }

    
    void set_alpha(float alpha_){alpha = alpha_;}
    
    void optimize(){
                        
        for(int32_t layerNum=0; layerNum<totalNumLayers; layerNum++){
            
            (*deltaWeights)[layerNum] = learning_rate*(*deltaWeights)[layerNum];            
            (*deltaZ)[layerNum] = learning_rate*(*deltaZ)[layerNum];
            
            (*weights)[layerNum] -= (*deltaWeights)[layerNum] + (learning_rate * alpha )* deltaWeights_old[layerNum];
            (*bias)[layerNum] -=  (*deltaZ)[layerNum].colwise().sum() + (learning_rate * alpha) * deltaZ_old[layerNum].colwise().sum();
        }
        
        copy_all_new2old();
    }
    
    void copy_all_new2old() {
        
        deltaZ_old = (*deltaZ);
        deltaWeights_old = (*deltaWeights);
    }

};
