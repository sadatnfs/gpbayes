Fitting Multidimensional Gaussian Process Regressions
============================================


[![Build Status](https://travis-ci.org/sadatnfs/gpbayes.svg?branch=master)](https://travis-ci.org/sadatnfs/gpbayes)


Contents include:
- Simulating multidimensional data across location, age, and time
- Fitting GPs in GPFlow and TMB
- Testing scalability for larger dimensional models



### Underlying Model to estimate:
```
Y_{c,a,t} = X^\beta + \Sigma  
\Sigma ~ GP({c} \kron {a} \kron {t})
```

### Current implementations:

- [GPFlow and ODVGP](https://github.com/sadatnfs/gpbayes/blob/master/scripts/01_GPFlow_ODVGP.py)
- [TMB](https://github.com/sadatnfs/gpbayes/blob/master/scripts/02_multilevel_GPR_model.cpp)

### References:

* [GPFlow](https://gpflow.readthedocs.io/en/master/?badge=master)
* [ODVGP](https://github.com/hughsalimbeni/orth_decoupled_var_gps)
* [TMB](https://github.com/kaskr/adcomp)
