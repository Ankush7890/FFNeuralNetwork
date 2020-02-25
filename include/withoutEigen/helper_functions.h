#define EPSILON 1e-10

using std::vector;
typedef vector < double > vecD_;
typedef vector < vector < double > > vecVecD_;
typedef vector < vector < double* > > vecVecDpnt_;

/**
 * @file
 * 
 * @brief Various loss and activation functions which can be used in the neural network.
 * 
 * @details Choices for activation function.<br>
 * \b softMax <br>
 * \b ReLU <br>
 * \b tanh <br>
 * \b sigmoid <br>
 * \b linear <br>
 * Choices for loss function.<br>
 * \b crossEntropy <br>
 * \b meanSquare <br>
 **/

/**************************************************/
//definition of crossEntropy function in all ways
/**************************************************/

double crossEntropy(vecVecD_ &y, vecVecD_ &y_pred)
{
    double loss=0;
    for(int32_t i=0;i<y.size();i++)
        for(int32_t j=0;j<y[i].size();j++)
            loss += -y[i][j]*log(y_pred[i][j]+EPSILON);
        
    return loss/y.size();
}

double crossEntropy(vecVecDpnt_ &y, vecVecD_ &y_pred)
{
    double loss=0;
    for(int32_t i=0;i<y.size();i++){
        for(int32_t j=0;j<y[i].size();j++)
            loss += -(*y[i][j])*log(y_pred[i][j]+EPSILON);        
    }
    return loss/y.size();
}

/**************************************************/
//definition of derivative of crossEntropy function 
//in all ways
/**************************************************/

void DcrossEntropy(vecVecD_ &y, vecVecD_ &y_pred, vecVecD_ &result)
{
    for(int32_t i=0;i<y.size();i++)
        for(int32_t j=0;j<y[i].size();j++)
            result[i][j] += -y[i][j]/(EPSILON+y_pred[i][j]);
}

vecVecD_ DcrossEntropy(vecVecD_ &y, vecVecD_ &y_pred)
{ /***changes required**/
    vecVecD_ result(y.size(),vecD_(y[0].size()));
    
    for(int32_t i=0;i<y.size();i++){
        for(int32_t j=0;j<y[i].size();j++)
            result[i][j] += -y[i][j]/(EPSILON+y_pred[i][j]);
    }
    return result;
}

vecVecD_ DcrossEntropy(vecVecDpnt_ &y, vecVecD_ &y_pred)
{
    
    vecVecD_ result(y.size(),vecD_(y[0].size()));
    
    for(int32_t i=0;i<y.size();i++){
        for(int32_t j=0;j<y[i].size();j++)
            result[i][j] += -(*y[i][j])/(EPSILON+y_pred[i][j]);
    }
    
    return result;
}

/**************************************************/
//definition of mean square function
/**************************************************/

double meanSquare(vecVecD_ &y, vecVecD_ &y_pred)
{
    double loss=0;
    for(int32_t i=0;i<y.size();i++)
        for(int32_t j=0;j<y[i].size();j++)
            loss += (y_pred[i][j]-y[i][j])*(y_pred[i][j]-y[i][j]);
        
    return loss;
}

double meanSquare(vecVecDpnt_ &y, vecVecD_ &y_pred)
{
    double loss=0;
    for(int32_t i=0;i<y.size();i++)
        for(int32_t j=0;j<y[i].size();j++)
            loss += (y_pred[i][j]-*y[i][j])*(y_pred[i][j]-*y[i][j]);
        
    return loss;
}

/**************************************************/
//definition of derivative of mean square function
/**************************************************/

vecVecD_ DmeanSquare(vecVecD_ &y, vecVecD_ &y_pred)
{
    vecVecD_ result(y.size(),vecD_(y[0].size()));
    for(int32_t i=0;i<y.size();i++)
        for(int32_t j=0;j<y[i].size();j++)
            result[i][j] += 2*(y_pred[i][j]-y[i][j]);
    return result;
}

void DmeanSquare(vecVecD_ &y, vecVecD_ &y_pred, vecVecD_ &result)
{
    for(int32_t i=0;i<y.size();i++)
        for(int32_t j=0;j<y[i].size();j++)
            result[i][j] += 2*(y_pred[i][j]-y[i][j]);
}

/**************************************************/
//definition of sigmoid function
/**************************************************/

void sigmoid(vecD_ &x, vecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++)
        result[i] = 1.0 / ( 1.0 + exp ( -x[i]) );
}

void sigmoid(vecVecD_ &x, vecVecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++)
            for(int32_t j=0; j<x[i].size(); j++)
                result[i][j] = 1.0 / ( 1.0 + exp ( -x[i][j]) );
}

/**************************************************/
//definition of derivative of sigmoid function
/**************************************************/

void dSigmoid(vecVecD_ &preFac, vecVecD_ &x, vecVecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++)
            for(int32_t j=0; j<x[i].size(); j++)
                result[i][j] = preFac[i][j]*x[i][j]*(1-x[i][j]);
}

void dSigmoid(vecD_ &y, vecD_ &result)
{
    for(int32_t i=0; i<y.size(); i++)
        result[i] = y[i]*(1-y[i]);
}

/**************************************************/
//definition of softMax function
/**************************************************/

void softMax(vecVecD_ &x, vecVecD_ &result)
{
    vecD_ sumEachRow(x.size(),0);
    
    for(int32_t i=0; i<x.size(); i++)
        for(int32_t j=0; j<x[0].size(); j++)
            sumEachRow[i] += exp(x[i][j]);
        
    for(int32_t i=0;i<x.size();i++)
        for(int32_t j=0;j<x[0].size();j++)
            result[i][j] = exp(x[i][j])/sumEachRow[i];
}

/**************************************************/
//definition of derivative of softMax function
/**************************************************/

void dSoftMax(vecVecD_ &preFac, vecVecD_ &x, vecVecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++){
        for(int32_t j=0; j<x[0].size(); j++){
            
            result[i][j]=0;
            for(int32_t k=0; k<x[0].size(); k++)
            result[i][j] += x[i][j]*((j==k) - x[i][k])*preFac[i][k];}}
}

/**************************************************/
//definition of tanh function	
/**************************************************/

void tanh(vecD_ &x, vecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++){
        
        double t  = exp(x[i]);
        double it = 1.0/t;
        result[i]=(t-it)/(t+it);
    }
}

//definition of tanh function
void tanh(vecVecD_ &x, vecVecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++)
            for(int32_t j=0; j<x[i].size(); j++){
                double t  = exp(x[i][j]);
                double it = 1.0/t;
                result[i][j]=(t-it)/(t+it);}
}

/**************************************************/
//definition of derivative of tanh function
/**************************************************/

void dTanh(vecVecD_ &preFac, vecVecD_ &x, vecVecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++)
            for(int32_t j=0; j<x[i].size(); j++)
                result[i][j]=preFac[i][j]*(1-x[i][j]*x[i][j]);
}

void dTanh(vecD_ &y, vecD_ &result)
{
    for(int32_t i=0;i<y.size();i++)
        result[i]=(1-y[i]*y[i]);
}
/**************************************************/
//definition of ReLU function	
/**************************************************/

void ReLU(vecD_ &x, vecD_ &result)
{
    for(int32_t i=0;i<x.size();i++)
        result[i] = x[i]>0?x[i]:0.01*x[i];
}

void ReLU(vecVecD_ &x, vecVecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++)
            for(int32_t j=0; j<x[i].size(); j++)
                result[i][j]=x[i][j]>0?x[i][j]:0.01*x[i][j];
}

/**************************************************/
//definition of derivative of ReLu function
/**************************************************/

void dReLU(vecVecD_ &preFac, vecVecD_ &x, vecVecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++)
            for(int32_t j=0; j<x[i].size(); j++)
                result[i][j]=preFac[i][j]*(x[i][j]>0?1:0.01);
}

void dReLU(vecD_ &y, vecD_ &result)
{
    for(int32_t i=0;i<y.size();i++)
        result[i] = y[i]>0?1.:0.01;
}

/**************************************************/
//definition of linear function	
/**************************************************/

void linear(vecD_ &x, vecD_ &result)
{
    for(int32_t i=0;i<x.size();i++)
        result[i] = x[i];
}

void linear(vecVecD_ &x, vecVecD_ &result)
{
    for(int32_t i=0; i<x.size(); i++)
            for(int32_t j=0; j<x[i].size(); j++)
                result[i][j]=x[i][j];
}

/**************************************************/
//definition derivative of linear function
/**************************************************/

void dLinear(vecVecD_ &preFac,vecVecD_ &x, vecVecD_ &result)
{
    result = preFac;
}

void dLinear(vecD_ &y, vecD_ &result)
{
    for(int32_t i=0;i<y.size();i++)
        result[i] = 1.;
}
