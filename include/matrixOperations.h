using std::vector;

typedef vector < double > vecD_;
typedef vector < vector < double > > vecVecD_;
typedef vector < vector < double* > > vecVecDpnt_;

/**
 * @file 
 * 
 * @section DESCRIPTION
 *
 * @brief functions for various matrix operations. 
 * 
 **/

/**************************************************/
//mat mulplication in all ways possible
/**************************************************/

void matMul(vecVecD_ &A, vecVecD_ &B, vecVecD_ &result){
    
    for(int32_t rowsA=0;rowsA<A.size();rowsA++)
        for(int32_t columnsB=0;columnsB<B[0].size();columnsB++){
            result[rowsA][columnsB]=0;
            for(int32_t common_RC=0;common_RC<B.size();common_RC++)
                result[rowsA][columnsB] += A[rowsA][common_RC]*B[common_RC][columnsB];
             }
 }

void matMul(float &c, vecVecD_ &B, vecVecD_ &result){
    
    for(int32_t rows=0;rows<B.size();rows++)
        for(int32_t columns=0;columns<B[0].size();columns++)
            result[rows][columns]= c*B[rows][columns];
        
 }
  
vecVecD_ matMul(vecVecD_ &A, vecVecD_ &B){
    
    vecVecD_ result(A.size(),vecD_(B[0].size()));
  
    for(int32_t rowsA=0;rowsA<A.size();rowsA++)
        for(int32_t columnsB=0;columnsB<B[0].size();columnsB++){
            result[rowsA][columnsB]=0;
            for(int32_t common_RC=0;common_RC<B.size();common_RC++)
                result[rowsA][columnsB] += A[rowsA][common_RC]*B[common_RC][columnsB];
             }
    return result;
 }
 
vecVecD_ matMul(vecVecDpnt_ &A, vecVecD_ B){
    
    vecVecD_ result(A.size(),vecD_(B[0].size()));
  
    for(int32_t rowsA=0;rowsA<A.size();rowsA++)
        for(int32_t columnsB=0;columnsB<B[0].size();columnsB++){
            result[rowsA][columnsB]=0;
            for(int32_t common_RC=0;common_RC<B.size();common_RC++)
                result[rowsA][columnsB] += *A[rowsA][common_RC] * (B[common_RC][columnsB]);
             }
    return result;
 }
 
/**************************************************/
//mat subtraction in all ways possible
/**************************************************/

void matSub(vecVecD_ &A, vecVecD_ &B, vecVecD_ &result){
    
    for(int32_t rows=0;rows<B.size();rows++)
        for(int32_t columns=0;columns<B[0].size();columns++)
            result[rows][columns]= A[rows][columns]-B[rows][columns];
        
 }

void matSub(vecD_ &A, vecVecD_ &B, vecD_ &result){
    
    vecD_ inter_result(A.size(),0);
    for(int32_t column=0; column<B[0].size(); column++){

        for(int32_t row=0;row<B.size();row++)
            inter_result[column] += B[row][column];
        
        result[column] = A[column]-inter_result[column];
    }
    
 }

vecD_ matSub(vecD_ &A, vecVecD_ &B){
    
    vecD_ result(A.size(),0);
    
    for(int32_t column=0; column<B[0].size(); column++){
       
        for(int32_t row=0;row<B.size();row++)
            result[column] += B[row][column];
        
        result[column] = A[column]-result[column];
    }
    return result;
 }
 
/**************************************************/
//mat transpose in all ways possible
/**************************************************/

vecVecD_ matTrans(vecD_ &A){
    
    vecVecD_ result;
    for(int32_t i=0;i<A.size();i++)
        result.push_back(vecD_(1,A[i]));
    
    return result;
}

vecVecD_ matTrans(vecVecD_ &A){
    
    vecVecD_ result(A[0].size());
    for(int32_t column=0;column<A[0].size();column++)
        for(int32_t row=0;row<A.size();row++)
            result[column].push_back(A[row][column]);
    
    return result;
}

vecVecD_ matTrans(vecVecDpnt_ A){
    
    vecVecD_ result(A[0].size());
    for(int32_t column=0;column<A[0].size();column++)
        for(int32_t row=0;row<A.size();row++)
            result[column].push_back(*A[row][column]);
    
    return result;
}

/**************************************************/
//mat elementwise in all ways possible
/**************************************************/

void matElementwise(vecD_ &A, vecD_ &B){
        
    for(int32_t i=0;i<B.size();i++)
        B[i] = A[i]*B[i];
}

void matElementwise(vecD_ &A, vecD_ &B, vecD_ &result){
        
    for(int32_t i=0;i<A.size();i++)
        result[i] = A[i]*B[i];
}

void matElementwise(vecVecD_ &A, vecVecD_ &B, vecVecD_ &result){
        
    for(int32_t i=0;i<A.size();i++)
        for(int32_t j=0;j<A[0].size();j++)
            result[i][j] = A[i][j]*B[i][j];
}
 
vecVecD_ matElementwise(vecVecD_ &A, vecVecD_ &B){
    
    vecVecD_ result(A.size());
    
    for(int32_t i=0;i<A.size();i++)
        for(int32_t k=0;k<B[0].size();k=k+A[0].size()){
            double sum=0;
            
            for(int32_t j=0;j<A[0].size();j++)
                sum += A[i][j]*B[i][j+k];
            
            result[i].push_back(sum);}
   
        return result;
}

/**************************************************/
//mat Addition in all ways possible
/**************************************************/

void matAdd(vecVecD_ &A, vecVecD_ &B, vecVecD_ &result){
    
    for(int32_t rows=0;rows<A.size();rows++)
        for(int32_t columns=0;columns<A[0].size();columns++)
            result[rows][columns]=A[rows][columns]+B[rows][columns];
}

void matAdd(vecVecD_ &A, vecD_ &B, vecVecD_ &result){
    
    for(int32_t rows=0;rows<A.size();rows++)
        for(int32_t columns=0;columns<A[0].size();columns++)
            result[rows][columns]=A[rows][columns]+B[columns];
}
