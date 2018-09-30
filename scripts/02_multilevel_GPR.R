

########################################################################
#### Author: Nafis Sadat
#### Simulate and fit a multidimensional Gaussian process regression
########################################################################


rm(list = ls())
pacman::p_load(
  data.table, TMB, ggplot2, geostatsp, INLA, brinla, MASS,
  ar.matrix, mvtnorm, Matrix, sparseMVN
)


########################################################################
### (0) Simulate fake data
########################################################################


# Number of location (L), age (A) and time (Y) points
len_L <- 10
len_A <- 15
len_Y <- 30
L <- c(1:len_L)
A <- c(1:len_A)
Y <- c(1:len_Y)

# Create a grid of the unique elements of L, A and Y
sim_data <- data.table(expand.grid(L = L, A = A, Y = Y))

# Simulate some random fixed effects values
sim_data[, x1 := runif(.N, 0, 1)]
sim_data[, x2 := runif(.N, 1, 2)]

# True coefficients on fixed effects and data noise
beta_x1 <- 0.4
beta_x2 <- 0.3
noise_LL <- 0.7


## Random effects ##

# Age #
# Simulate AR1 random effect on age
ar1_phi_A <- 0.75
ar1_sigma_A <- 0.25

# Create sparse precision matrix of the age (AR1)
prec_A_ar1 <- Q.AR1(length(A), sigma = ar1_sigma_A, rho = ar1_phi_A, sparse = T)

# Location #
# Simulate IID random effect on location
sigma_L <- 0.3
prec_L_iid <- sparseMatrix(len_L, len_L)
diag(prec_L_iid) <- 1 / sigma_L**2

# Time #
# Matern covariance parameters
sigma0 <- .1 # Standard deviation
range0 <- .5 # Spatial range
kappa <- sqrt(8) / range0 # inla paramter transform
tau <- 1 / (sqrt(4 * pi) * kappa * sigma0) # inla parameter transform

# Simulate a Matern precision matrix based off of the time mesh
mesh_time <- inla.mesh.1d(c(1:len_Y))
spde_time <- inla.spde2.matern(mesh_time)
prec_Y_mat <- inla.spde2.precision(spde_time, theta = c(log(tau), log(kappa)))


# Create a multidimensional GP sparse precision matrix (using a kronecker)
Q_all <- kronecker(kronecker(prec_Y_mat, prec_A_ar1), prec_L_iid)

# Simulate based off of the sparse precision matrix
simm <- array(
  data = rmvn.sparse(1, rep(0, nrow(Q_all)), Cholesky(Matrix(Q_all, sparse = TRUE)), T),
  dim = c(length(L), length(A), length(Y))
)

# Melt the multidimensional array to be merged on to our original grid dataset
simm <- melt(simm)
colnames(simm) <- c("L", "A", "Y", "int")

sim_data <- merge(sim_data, simm, c("L", "A", "Y"))


## Simulate output with some IID noise ##
sim_data[, y_sim := int + beta_x1 * x1 + beta_x2 * x2 + rnorm(.N, sd = noise_LL)]



########################################################################
### (1) Compile TMB model
########################################################################

## This code will set the max number of OpenMP threads, for faster computation
dyn.load("/opt/compiled_code_for_R/setthreads.so")
invisible(.C("setOMPthreads", as.integer(40)))
invisible(.C("setMKLthreads", as.integer(40)))

## Clean, compile and load model
# system('rm ~/gpbayes_GH/scripts/*.o ~/gpbayes/scripts/*.so'); TMB::compile('~/gpbayes_GH/scripts/02_multilevel_GPR_model.cpp')
dyn.load(dynlib("~/gpbayes_GH/scripts/02_multilevel_GPR_model"))




########################################################################
### (2) Prep for running model
########################################################################


### DATA ARRAYS ###

## Fixed effects ##
array_x <- melt(sim_data[, .(L, A, Y, x1, x2)], c("L", "A", "Y"), variable.name = "C")
array_x[, C := as.numeric(C)]
array_x <- reshape2::acast(array_x, L ~ A ~ Y ~ C, value.var = "value")


## Yvar ##
array_y <- sim_data[, .(L, A, Y, y_sim)]
array_y <- reshape2::acast(array_y, L ~ A ~ Y, value.var = "y_sim")


## Time mesh for Matern GP ##
mesh_time <- inla.mesh.1d(sim_data$Y)
spde_time <- inla.spde2.matern(mesh_time)
M0_mat <- spde_time$param.inla$M0
M1_mat <- spde_time$param.inla$M1
M2_mat <- spde_time$param.inla$M2


## Create list of data
data_in <- list(
  X = array_x, Y = array_y,
  M0 = M0_mat, M1 = M1_mat, M2 = M2_mat
)



### PARMATERS ###

## Initialize all the parameters with proper shapes
parm_in <- list(
  param_b = c(0, 0),
  param_logSigma = 0.,
  Epsilon_stz = array(0, dim(data_in$Y)),
  param_L_sigma = 0.,
  param_A = 0.,
  param_A_sigma = 0.,
  logtau = 0.,
  logkappa = 0.
)






########################################################################
### (3) Run model in TMB
########################################################################


## A list with all model inputs
output_list <- list()

### Make AutoDiff function
output_list[["obj"]] <- MakeADFun(data_in,
  parm_in,
  hessian = T,
  silent = F,
  random = c("Epsilon_stz"),
  checkParameterOrder = T,
  DLL = "05_multilevel_GPR_model",
  last.par = T
)

## Optimize and time it
system.time(output_list[["opt"]] <- nlminb(
  objective = output_list[["obj"]]$fn,
  gradient = output_list[["obj"]]$gr,
  start = output_list[["obj"]]$par
))


## Look at the params we just fit
output_list[["obj"]]$report()


## Get the SEs of the parameter estimates (may return NaN for repeated random effects)
output_list[["FC_model"]] <- sdreport(output_list[["obj"]],
  getJointPrecision = T,
  getReportCovariance = T,
  bias.correct = F
)

## Look at our predictions, and record it to the dataframe
Yhat_pred <- output_list[["obj"]]$report()$Yhat
dimnames(Yhat_pred) <- dimnames(array_y)
Yhat_pred <- data.table(melt(Yhat_pred))
colnames(Yhat_pred) <- c("L", "A", "Y", "Yhat_pred")
sim_data <- merge(sim_data, Yhat_pred, c("L", "A", "Y"))


## Make some plots
ggplot(sim_data[L == 1]) +
  geom_point(aes(x = Y, y = y_sim)) +
  geom_line(aes(x = Y, y = Yhat_pred)) +
  facet_wrap(~A)





########################################################################
### (4) Simulate draws of the parameters, and create draws of Yhat
########################################################################

rmvnorm_prec <- function(mu, prec, n.sims) {
  z <- matrix(rnorm(length(mu) * n.sims), ncol = n.sims)
  L <- Cholesky(prec)
  z <- solve(L, z, system = "Lt") ## z = Lt^-1 %*% z
  z <- solve(L, z, system = "Pt") ## z = Pt    %*% z
  z <- as.matrix(z)
  return(mu + z)
}


## Pull out precision matrix and mean
mu <- c(output_list[["FC_model"]]$par.fixed, 
        output_list[["FC_model"]]$par.random)

# Simulate draws
samples <- 100
draws <- rmvnorm_prec(mu = mu * 0, 
                      output_list[["FC_model"]]$jointPrecision, samples)


## Separate out the draws
parnames <- c(names(output_list[["FC_model"]]$par.fixed), 
              names(output_list[["FC_model"]]$par.random))
epsilon_draws <- draws[parnames == "Epsilon_stz", ]
FE_draws <- draws[parnames == "param_b", ]

## Make predictions of Yhat
draws_yhat <- rbindlist(lapply(c(1:samples), function(d) {
  FE_fit <- (sim_data$x1 * FE_draws[1, d]) + (sim_data$x2 * FE_draws[2, d])
  GP_fit <- epsilon_draws[, d]
  Yfit <- data.table(draw = d, Ypred = GP_fit + FE_fit)

  ## Add metadata of L,A,Y
  Yfit <- cbind(sim_data[, .(L, A, Y)], Yfit)
  return(Yfit)
}))



## Create 95% quantiles of the draws
quantile_Yhat <- draws_yhat[, as.list(c(mean(Ypred), 
                                        quantile(Ypred, c(0.025, 0.975)))), 
                            by = c("L", "A", "Y")]
colnames(quantile_Yhat) <- c("L", "A", "Y", "mean", "lower", "upper")



## Make some plots (for a given location, across time and age)
Lplot <- 10
ggplot() +

  ## Fitted
  geom_ribbon(data = quantile_Yhat[L == Lplot], 
              aes(x = Y, ymin = lower, ymax = upper), 
              alpha = 0.3, color = NA, fill = "steelblue") +
  geom_line(data = quantile_Yhat[L == Lplot], aes(x = Y, y = mean)) +
  ## Raw
  geom_point(data = sim_data[L == Lplot], aes(x = Y, y = y_sim)) +

  facet_wrap(~A)

## Plot for a given age, across time and location
Aplot <- 3
ggplot() +

  ## Fitted
  geom_ribbon(data = quantile_Yhat[A == Aplot], 
              aes(x = Y, ymin = lower, ymax = upper), 
              alpha = 0.3, color = NA, fill = "steelblue") +
  geom_line(data = quantile_Yhat[A == Aplot], aes(x = Y, y = mean)) +
  ## Raw
  geom_point(data = sim_data[A == Aplot], aes(x = Y, y = y_sim)) +

  facet_wrap(~L)
