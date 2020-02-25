#ifndef HELPERFUNCTION_H
#define HELPERFUNCTION_H

#define EPSILON 1e-10

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <memory>

using std::vector;
using std::string;
using std::map;

using Eigen::MatrixXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/**
 * @file 
 * 
 * @brief Various loss and activation functions which can be used in the neural network.
 * 
 * @details Choices for activation function.<br>
 * \b SoftMax <br>
 * \b ReLU <br>
 * \b Tanh <br>
 * \b Sigmoid <br>
 * \b Linear <br>
 * 
 * Choices for loss function.<br>
 * \b CrossEntropy <br>
 * \b MeanSquare <br>
 * \b FocalLoss <br>
 * \b FocalLoss_b <br>
 * 
 * FocalLoss_b is the bad version of the original focal loss function {arXiv:1708.02002}. This was mistakenly implemented in the 
 * first implementation but still gives some interesting results for implicit solvent training. But it has not been well tested.
 * 
 **/

/**
 *
 * @class BaseFn
 *
 * @brief BaseFn class for all the loss function and activation functions.
 * 
 * @details Base functions.<br>
 * \b func Activation function. <br>
 * \b derivative (with void return) derivative of Activation function. <br>
 * \b loss_f loss function.<br>
 * \b derivative (with double return) derivate of loss function.<br>
 * \b set_alpha_gamma setter for alpha and gamma value of focal loss function.<br>
 * 
 **/

struct BaseFn{
    
    BaseFn(){};
    virtual void error_call(){throw std::runtime_error("Base Function method should not be used!\n");};
    virtual void func(ArrayXXd &x, ArrayXXd &result){error_call();};
    virtual void derivative(ArrayXXd &preFac, ArrayXXd &x, ArrayXXd &result){error_call();};

    virtual double loss_f(ArrayXXd &x, ArrayXXd &result){error_call();};
    virtual ArrayXXd derivative(ArrayXXd &y, ArrayXXd &y_pred){error_call();};
    virtual void set_alpha_gamma(float alpha_, float gamma_){error_call();};
    
};


/**ActivationFunctions**/

struct Sigmoid: public BaseFn{
    
    void func(ArrayXXd &x, ArrayXXd &result){
        
        result = 1.0 / ( 1.0 + exp(-x) );
    }
    
    void derivative(ArrayXXd &preFac, ArrayXXd &x, ArrayXXd &result){
        ArrayXXd pf1 = x*(1-x);
        result = preFac*pf1;
    }
    
};



struct SoftMax: public BaseFn{
    
    void func(ArrayXXd &x, ArrayXXd &result){
        
        ArrayXd sumEachRow = exp(x).rowwise().sum();
        result = exp(x).colwise() / sumEachRow;
    }

    
    void derivative(ArrayXXd &preFac, ArrayXXd &x, ArrayXXd &result){
        
        ArrayXXd Iden = MatrixXd::Identity(x.cols(), x.cols());
        
        for(int32_t eg=0; eg<x.rows(); eg++){
            
            ArrayXXd featM = Iden.rowwise()*x.row(eg) - (x.row(eg).matrix().transpose()*x.matrix().row(eg)).array();
            featM = featM.rowwise()*preFac.row(eg);
            result.row(eg) = featM.rowwise().sum().transpose();
        }
    }

};

struct Tanh: public BaseFn{

    void func(ArrayXXd &x, ArrayXXd &result){
        
        result = tanh(x);}

    
    void derivative(ArrayXXd &preFac, ArrayXXd &x, ArrayXXd &result){
        
        result = preFac*(1-x*x);}
    
};


struct ReLU: public BaseFn{
    
    void func(ArrayXXd &x, ArrayXXd &result){
        
        ArrayXXd tf1 = (x>0).cast<double>();
        ArrayXXd tf2 = (x<0).cast<double>();
        result = tf1*x + 0.01*tf2*x;}

    
    void derivative(ArrayXXd &preFac, ArrayXXd &x, ArrayXXd &result){
        
        ArrayXXd tf1 = (x>0).cast<double>();
        tf1 += 0.01*(x<0).cast<double>();
        result = preFac*tf1;
    }

};

struct Linear: public BaseFn{
    
    void func(ArrayXXd &x, ArrayXXd &result){
        
        result = x;}

    
    void derivative(ArrayXXd &preFac, ArrayXXd &x, ArrayXXd &result){
        ArrayXXd pf1 = 0*x.array()+1;
        result = preFac*pf1;
    }
    
};

/**LossFunctions**/

struct MeanSquare: public BaseFn{
    
    double loss_f(ArrayXXd &y, ArrayXXd &y_pred){
        
        double loss = ((y_pred-y)*(y_pred-y)).sum();
        return loss/y.rows();}


    ArrayXXd derivative(ArrayXXd &y, ArrayXXd &y_pred){
        
        ArrayXXd result(y.rows(),y.cols());
        result = 2*(y_pred-y);
        return result;}


};


struct CrossEntropy: public BaseFn{
    double loss_f(ArrayXXd &y, ArrayXXd &y_pred){
        
        double loss = (-y*log(y_pred+EPSILON)).sum();
        return loss/y.rows();}

    ArrayXXd derivative(ArrayXXd &y, ArrayXXd &y_pred){
        
        ArrayXXd result(y.rows(),y.cols());    
        result = -y/(EPSILON+y_pred);
        return result;}
};



struct FocalLoss: public BaseFn{
    
    float alpha_l,gamma_l;
    
    void set_alpha_gamma(float alpha_, float gamma_){
        
        alpha_l=alpha_;
        gamma_l=gamma_;}
    
    ArrayXXd crossEntropywithLogits(ArrayXXd &y, ArrayXXd &y_pred){
        
        ArrayXXd logits = -y*log(y_pred+EPSILON)-(1-y)*log(1-y_pred+EPSILON);
        return logits;}
        
    ArrayXXd DcrossEntropywithLogits(ArrayXXd &y, ArrayXXd &y_pred){
        
        ArrayXXd logits = -y/(y_pred+EPSILON) + (1-y)/(1-y_pred+EPSILON);
        return logits;}

    
    double loss_f(ArrayXXd &y, ArrayXXd &y_pred){
        
        ArrayXXd ce = crossEntropywithLogits(y, y_pred);
        ArrayXXd pt = y_pred*y + (1-y_pred)*(1-y);
        
        ArrayXXd alphaFact = alpha_l*y + (1-alpha_l)*(1-y);
        ArrayXXd gammaFact = pow(1-pt, gamma_l);
        
        ArrayXXd finalReturn = alphaFact * gammaFact * ce;
        
        return finalReturn.sum()/y.rows();}

    ArrayXXd derivative(ArrayXXd &y, ArrayXXd &y_pred){
        
        ArrayXXd result(y.rows(),y.cols());    
        ArrayXXd dce = DcrossEntropywithLogits(y, y_pred);
        ArrayXXd pt = y_pred*y + (1-y_pred)*(1-y);
        
        ArrayXXd alphaFact = alpha_l*y + (1-alpha_l)*(1-y);
        ArrayXXd gammaFact = gamma_l * pow(1-pt, gamma_l-1) *(2*y - 1);
        
        return alphaFact*gammaFact*dce;}
};


struct FocalLoss_b: public BaseFn{
    
    float alpha_l,gamma_l;
    
    void set_alpha_gamma(float alpha_, float gamma_){
        
        alpha_l=alpha_;
        gamma_l=gamma_;}
    
    double loss_f(ArrayXXd &y, ArrayXXd &y_pred){
        
        ArrayXXd y_pow = pow((1.0-y_pred), gamma_l);
        double loss = (-alpha_l*y*y_pow*log(y_pred+EPSILON)).sum();
        return loss/y.rows();
    }

    ArrayXXd derivative(ArrayXXd &y, ArrayXXd &y_pred){
        
        ArrayXXd result(y.rows(),y.cols());    
        ArrayXXd y_pow_1 = pow(1.0-y_pred, gamma_l-1);
        ArrayXXd y_pow = y_pow_1*(1.0-y_pred);
        
        result = gamma_l*y*y_pow_1*log(y_pred+EPSILON) - y_pow*y/(y_pred+EPSILON);
        result *= alpha_l;
        
        return result;}

};


#endif	 

