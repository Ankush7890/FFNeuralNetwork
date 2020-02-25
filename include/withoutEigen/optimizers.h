using std::vector;
typedef vector < double > vecD_;
typedef vector < vector < double > > vecVecD_;
typedef vector < vector < double* > > vecVecDpnt_;

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

struct BaseOptimizer{
    
    //constructor
    BaseOptimizer(){};
    virtual void optimize(){};
    virtual void set_alpha(float alpha){};

};

struct GradientDescent : public BaseOptimizer
{
//     virtual ~GradientDescent();
    float learning_rate, totalNumLayers;
    vector < vecVecD_ >* weights;
    vecVecD_* bias;
    
    vector < vecVecD_ >* deltaWeights;
    vector < vecVecD_ >* deltaZ;
        
    //constructor
    GradientDescent(float learning_rate_, vector < vecVecD_ >* weights_, vecVecD_* bias_,  vector < vecVecD_ >* deltaWeights_, vector < vecVecD_ >* deltaZ_)
    :learning_rate(learning_rate_),
    weights(weights_),
    bias(bias_),
    deltaWeights(deltaWeights_),
    deltaZ(deltaZ_)
    {
        totalNumLayers =(*weights).size();
    };
    
    void optimize(){
        for(int32_t laynum=0; laynum<totalNumLayers; laynum++){
            matMul(learning_rate, (*deltaWeights)[laynum], (*deltaWeights)[laynum]);        
            matMul(learning_rate, (*deltaZ)[laynum], (*deltaZ)[laynum]);
            matSub((*weights)[laynum], (*deltaWeights)[laynum], (*weights)[laynum]);
            matSub((*bias)[laynum], (*deltaZ)[laynum], (*bias)[laynum]);
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
    
    vector < vecVecD_ > deltaWeights_old;
    vector < vecVecD_ > deltaZ_old;
    bool inoptimize=false;
    
    //constructor
    GradientDescentWithMomentum(float learning_rate_, vector < vecVecD_ >* weights_, vecVecD_* bias_,  vector < vecVecD_ >* deltaWeights_, vector < vecVecD_ >* deltaZ_)
    :GradientDescent(learning_rate_, weights_, bias_, deltaWeights_, deltaZ_)
    {
        copy_all_new2old();        
    }

    void set_alpha(float alpha_){alpha = alpha_;}
    
    void optimize(){
        
        if(inoptimize==false)
            copy_all_new2old();
        
        inoptimize=true;
        
        for(int32_t laynum=0; laynum<totalNumLayers; laynum++){
            
            //learning_rate*deltaW
            matMul(learning_rate, (*deltaWeights)[laynum], (*deltaWeights)[laynum]);        
            
            //learning_rate*deltaB            
            matMul(learning_rate, (*deltaZ)[laynum], (*deltaZ)[laynum]);
            
            //W = W - learning_rate*deltaW            
            matSub((*weights)[laynum], (*deltaWeights)[laynum], (*weights)[laynum]);
            
            //B = B - learning_rate*deltaB                        
            matSub((*bias)[laynum], (*deltaZ)[laynum], (*bias)[laynum]);

            //alpha*deltaW(t-1)
            matMul(alpha, deltaWeights_old[laynum], deltaWeights_old[laynum]);        

            //alpha*deltaB(t-1)            
            matMul(alpha, deltaZ_old[laynum], deltaZ_old[laynum]);

            //W = W - alpha*(learning_rate*deltaW{at time t-1})            
            matSub((*weights)[laynum], deltaWeights_old[laynum], (*weights)[laynum]);

            //B = B - alpha*(learning_rate*deltaB{at time t-1})            
            matSub((*bias)[laynum], deltaZ_old[laynum], (*bias)[laynum]);

        }
        copy_all_new2old();
    }
    
    void copy_all_new2old() {
        deltaZ_old = (*deltaZ);
        deltaWeights_old = (*deltaWeights);
    }

};
