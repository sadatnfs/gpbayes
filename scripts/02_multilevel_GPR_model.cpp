// ///////////////////////////////////////////////////////////////////////
// Nafis Sadat, Roy Burstein, Aaron Osgood-Zimmerman and Neal Marquez
// Created: September 2018
// Template file for fitting a location-age-time Gaussian process model
// ///////////////////////////////////////////////////////////////////////

// include libraries
#include <TMB.hpp>
#include <Eigen/Sparse>
#include <vector>
using namespace density;
using Eigen::SparseMatrix;

// NAME:spde_Q
// DESC: helper function to make sparse SPDE precision matrix
// Inputs:
//    logkappa: log(kappa) parameter value
//    logtau: log(tau) parameter value
//    M0, M1, M2: these sparse matrices are output from R::INLA::inla.spde2.matern()$param.inla$M*
template<class Type>
SparseMatrix<Type> spde_Q(Type logkappa, Type logtau, SparseMatrix<Type> M0,
                          SparseMatrix<Type> M1, SparseMatrix<Type> M2) {
  SparseMatrix<Type> Q;
  Type kappa2 = exp(2. * logkappa);
  Type kappa4 = kappa2*kappa2;
  Q = pow(exp(logtau), 2.)  * (kappa4*M0 + Type(2.0)*kappa2*M1 + M2);
  return Q;
}

// NAME:ar_Q
// DESC: helper function to make sparse precision matrix of an ar1 process
// Inputs:
//    N: number of elements in vector
//    rho: rho parameter value (autocorrelation)
//    sigma: variance of the AR1 process
template<class Type>
SparseMatrix<Type> ar_Q(int N, Type rho, Type sigma) {
  SparseMatrix<Type> Q(N,N);
  Q.insert(0,0) = (1.) / pow(sigma, 2.);
  for (size_t n = 1; n < N; n++) {
    Q.insert(n,n) = (1. + pow(rho, 2.)) / pow(sigma, 2.);
    Q.insert(n-1,n) = (-1. * rho) / pow(sigma, 2.);
    Q.insert(n,n-1) = (-1. * rho) / pow(sigma, 2.);
  }
  Q.coeffRef(N-1,N-1) = (1.) / pow(sigma, 2.);
  return Q;
}

// NAME:iid_Q
// DESC: helper function to make precision matrix of an IID process
// Inputs:
//    N: number of elements in vector
//    sigma: variance of the normal distribution
template<class Type>
SparseMatrix<Type> iid_Q(int N, Type sigma){
  SparseMatrix<Type> Q(N, N);
  for(int i = 0; i < N; i++){
    Q.insert(i,i) = 1. / pow(sigma, 2);
  }
  return Q;
}

// NAME: isNA
// DESC: Detect if a data point supplied from R is NA or not
// Inputs:
//    x: input vector
template<class Type>
bool isNA(Type x){
  return R_IsNA(asDouble(x));
}


// NAME: stationary_coerce
// DESC: maps a Type scalar into the (-1,1) range
// Inputs:
//    x: input scalar of type Type
template<class Type>
Type stationary_coerce(Type x){
  return (exp(x) - 1) / (exp(x) + 1); 
}


 

// Our objective function for TMB to optimize over
template<class Type>
  Type objective_function<Type>::operator() ()
{
  
  
  // Set max number of OpenMP threads to help us optimize faster
  max_parallel_regions = omp_get_max_threads();
    

  // Initialize nll
  Type nll = 0.;
  
  
  // Data setup //
  
    // Data array of Y (L~A~T)
    DATA_ARRAY(Y);
    
    // vector and data of fixed effects (L~A~T)
    PARAMETER_VECTOR(param_b);
    DATA_ARRAY(X);
    
    
    // Data likelihood noise
    PARAMETER(param_logSigma);
    
    // Create a placeholder for the predictions
    array<Type> Yhat(Y.dim);
    
    
  // Random effects setup //
    
    // Parameter holders (the actual random effects array we will estimate into)
    PARAMETER_ARRAY(Epsilon_stz);
      
      //// Location ////
        // Variances of location and age (coerced to positive values)
        PARAMETER(param_L_sigma);
        
        Type fit_param_L_sigma = exp(param_L_sigma);
        
        
      //// Age ////
        // AR1 correlations across age
        PARAMETER(param_A);
        
        // Make sure that we estimate staionary processes for the AR1 process (on age)
        Type param_A_tr    = stationary_coerce(param_A); 
  
        // Variance of the AR1 process on age
        PARAMETER(param_A_sigma);
        Type fit_param_A_sigma = exp(param_A_sigma);
            
      

      //// Time ////
      
        // Get the SPDE structure from R-INLA
        DATA_SPARSE_MATRIX(M0);    // used to make gmrf precision
        DATA_SPARSE_MATRIX(M1);    // used to make gmrf precision
        DATA_SPARSE_MATRIX(M2);    // used to make gmrf precision
        
        // Matern covariance parameters
        PARAMETER(logkappa);
        PARAMETER(logtau);
        
        
        
      // Add random effects to the negative log likelihood
      PARALLEL_REGION nll += SEPARABLE( GMRF( spde_Q( logkappa, logtau, M0, M1, M2) )   , 
                                        SEPARABLE( SCALE(AR1(param_A_tr),  fit_param_A_sigma/sqrt(Type(1.0)-param_A_tr*param_A_tr))  , 
                                                   GMRF( iid_Q( Y.dim(0), pow(fit_param_L_sigma, 2) ) )   ))(Epsilon_stz);
      
   
   // Report the random effects for sanity checks
     REPORT(param_A_tr);
     REPORT(logtau);
     REPORT(logkappa);
     REPORT(Epsilon_stz);
   
   
   
   
   // Predictions setup //
   
         // Add the random effects to Yhat
      for(int dim0=0; dim0 < Y.dim(0); dim0++) {
        for(int dim1=0; dim1 < Y.dim(1); dim1++) {
          for(int dim2=0; dim2 < Y.dim(2); dim2++) {
            Yhat(dim0, dim1, dim2) = Epsilon_stz(dim0, dim1, dim2);
          }
        }
      }
      
    // Add fixed effects
    for(int dim0=0; dim0 < Y.dim(0); dim0++) {
      for(int dim1=0; dim1 < Y.dim(1); dim1++) {
        for(int dim2=0; dim2 < Y.dim(2); dim2++) {
          for(int dim3=0; dim3 < X.dim(3); dim3++) {
          
            // Add to Yhat
            Yhat(dim0, dim1, dim2) += X(dim0, dim1, dim2, dim3) * param_b[dim3];
            
          }
        }
      }
    }
    
    
    // Add data likelihood to NLL
     for(int dim0=0; dim0 < Y.dim(0); dim0++) {
       for(int dim1=0; dim1 < Y.dim(1); dim1++) {
         for(int dim2=0; dim2 < Y.dim(2); dim2++) {
           
           if(!CppAD::isnan(Y(dim0, dim1, dim2)) & !isNA(Y(dim0, dim1, dim2))) {
             PARALLEL_REGION nll -= dnorm(Y(dim0, dim1, dim2), Yhat(dim0, dim1, dim2), exp(param_logSigma), true);
           }
         }
       }
     }
     
    // Return predictions
     REPORT(Yhat);
        
    return nll ;
  }



