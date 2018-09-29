import numpy as np
import pandas as pd

from scipy.cluster.vq import kmeans2
from bayesian_benchmarks.tasks.regression import run as run_regression
from gpflow.kernels import Matern52, Matern12, Linear
from gpflow.likelihoods import Gaussian
from gpflow.training import NatGradOptimizer, AdamOptimizer
from odvgp.odvgp import ODVGP, DVGP


class SETTINGS:
    # model
    likelihood_variance = 1e-2
    lengthscales = 0.1

    # training
    iterations = 500
    ng_stepsize = 5e-2
    adam_stepsize = 1e-3
    minibatch_size = 1000
    gamma_minibatch_size = 64


class Model_ODVGP:
    """
    Bayesian_benchmarks-compatible wrapper around the orthogonally decoupled variational GP from 

    @inproceedings{salimbeni2018decoupled,
      title={Orthogonally Decoupled Variational Gaussian Processes},
      author={Salimbeni, Hugh and Cheng, Ching-An and Boots, Byron and Deisenroth, Marc},
      booktitle={Advances in Neural Information Processing Systems},
      year={2018}
    }

    The natural gradient step for the beta parameters can be implemented using standard 
    gpflow tools, due to the decoupling. We optimize the rest of the parameters using adam. 

    """

    def __init__(self, M_gamma, M_beta, linear_dims, matern_dims):
        # define the number of each of the inducing points
        self.M_gamma = M_gamma
        self.M_beta = M_beta
        self.linear_dims = linear_dims
        self.matern_dims = matern_dims
        self.model = None

    def init_model(self, Model, X, Y):
        """
        Initialize the model
        TODO: Currently I'm coding in the choice of having purely a combo
        of Matern and Linear. We can make this flexible.
        """
        Dx = X.shape[1]
        kern = Matern52(input_dim=len(self.matern_dims),
                        active_dims=self.matern_dims,
                        lengthscales=SETTINGS.lengthscales * Dx ** 0.5) + \
            Linear(input_dim=len(self.linear_dims),
                   active_dims=self.linear_dims)
        lik = Gaussian()
        lik.variance = SETTINGS.likelihood_variance

        gamma = kmeans2(X, self.M_gamma, minit='points')[
            0] if self.M_gamma > 0 else np.empty((0, Dx))
        beta = kmeans2(X, self.M_beta, minit='points')[0]

        if self.M_gamma > 0:
            gamma_minibatch_size = SETTINGS.gamma_minibatch_size 
        else 
            gamma_minibatch_size = None

        self.model = Model(X, Y, kern, lik, gamma, beta,
                           minibatch_size=SETTINGS.minibatch_size,
                           gamma_minibatch_size=gamma_minibatch_size)
        self.sess = self.model.enquire_session()

    def fit(self, X, Y):
        """
        Optimize
        """
        if not self.model:
            self.init_model(ODVGP, X, Y)

        var_list = [[self.model.basis.a_beta, self.model.basis.L]]
        self.model.basis.a_beta.set_trainable(False)

        op_ng = NatGradOptimizer(SETTINGS.ng_stepsize).make_optimize_tensor(
            self.model, var_list=var_list)
        op_adam = AdamOptimizer(
            SETTINGS.adam_stepsize).make_optimize_tensor(self.model)
        for it in range(SETTINGS.iterations):
            self.sess.run(op_ng)
            self.sess.run(op_adam)

            if it % 50 == 0:
                print('Iter: {}, Loss:{:.4f}'.format(
                    it, self.sess.run(self.model.likelihood_tensor)))

        self.model.anchor(self.sess)

    def predict(self, Xs):
        """
        Get the fitted mean and covariance matrix
        TODO: function to simulate realizations of the process
        """
        return self.model.predict_y(Xs, session=self.sess)

    def predict_full_cov(self, Xs):
        return self.model.predict_f_full_cov(Xs, session=self.sess)


def create_data(len_L, len_A, len_T):
    """
    Simulate data, use just 2 fixed effects for now
    """
    L = [i for i in range(len_L)]
    A = [i for i in range(len_A)]
    T = [i for i in range(len_T)]

    # Sim data
    beta_x1 = 0.4
    beta_x2 = 0.3
    LL_noise = 0.25


    ## Create numpy array of datas
    def cartesian(*arrays):
        """
        Simulates multidimensional dataframe
        for each unique indices of data
        """
        mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
        dim = len(mesh)  # number of dimensions
        elements = mesh[0].size  # number of elements, any index will do
        flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
        reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
        return reshape


    full_data = cartesian(L, A, T)
    full_data = np.hstack([full_data, np.random.uniform(size = [full_data.shape[0],1]), \
                      np.random.normal(size = [full_data.shape[0],1])])
    full_data = np.hstack([full_data, (beta_x1*full_data[:,3] + beta_x2*full_data[:,4])[:,None] ])

    return full_data


def train_test_split(data):
    ## Create train-test data splits (drop every 2 years)
    data_train = data[::2]
    data_test = data

    X_train = data_train[:,0:5] ## Be careful of the indices here!!!!!
    Y_train = data_train[:,5][:,None]

    X_test = data_test[:,0:5]
    Y_test = data_test[:,5][:,None]

    return X_train, Y_train, X_test, Y_test


def main():
    ## Initialize model with x,y partitions on gamma and beta
    model = Model_ODVGP(150, 100, linear_dims=[3,4], matern_dims=[0,1,2])

    ## Make data
    data_sim = create_data(20,10,20)
    X_train, Y_train, X_test, Y_test = train_test_split(data_sim)

    model.fit(X_train, Y_train)
    m, v = model.predict_full_cov(X_test)

    ## Look at fits
    predictions = np.hstack([full_data, m])
    pd.DataFrame(predictions)

if __name__ == '__main__':
    main()

