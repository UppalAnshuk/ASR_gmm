"""Author : Anshuk Uppal (IIITB,IMT2015503)
Comments: This is the main GMM module which states the model and the expectation maximization training
algorithm. This is inspired by following-
https://github.com/ldeecke/gmm-torch
https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html
"""
import torch as tr
from torch import nn
import numpy as np
from math import pi

class GaussianMixtureModel(nn.module):
    """Assuming a m component GMM , INPUT is n x f (num_samples x num_features),mu and sigma are the
    mean and covariance matrices each of dimension (1,m,f)"""
    def __init__(self,num_components,num_features,mean_init=None,var_init=None,epsilon=1e-6):
        """
        x:Flatted input (n,f)
        z:Latent variable which represents mixing coefficients (1,m,1) for all components
        :param num_components: m(int)
        :param num_features: f(int)
        :param mean_init: tr.tensor(1,m,f)
        :param var_init: tr.tensor(1,m,f)
        :param epsilon: float, for avoiding divide by zero
        """
        super(GaussianMixtureModel,self).__init__()

        self.eps=epsilon
        self.num_components=num_components
        self.num_features=num_features
        self.likelihood=-np.inf #log likelihood of all the data points given the model

        if mean_init is not None:
            assert mean_init.size() == (
            1, num_components, num_features), "Unmatched tensor dimensions: expected (1, %i, %i) got (%i)" % (
            num_components, num_features,str(mean_init.shape))

            self.mean=tr.nn.Parameter(mean_init,requires_grad=False) #no need of storing gradients as EM is used for training
        else:
            self.mean=tr.nn.Parameter(tr.randn(1,num_components,num_features),requires_grad=False)
        if var_init is not None:
            assert var_init.size() == (
                1, num_components, num_features), "Unmatched tensor dimensions: expected (1, %i, %i) got (%i)" % (
                num_components, num_features, str(var_init.shape))
            self.var=tr.nn.Parameter(var_init,requires_grad=False)
        else:
            self.var=tr.nn.Parameter(tr.random.randn(1,num_components,num_features),requires_grad=False)

        self.z=tr.empty(1,num_components,1)
        self.z.data=(tr.ones(1,num_components,1))/num_components #equally likely mixture components

    def fit(self,x,epochs=200,dl=1e-5):
        """Fitting the data to the given mixture model.
        As the algorithm proceeds we expect the log likelihood of the data to increase.
        Maximising likelihood through Expectation maximization(Hill climbing)"""

        if len(x.shape)==2:
    #       n x f to n x m x f, broadcasting the data along a 3rd dimension for simplifying
    #       tensor computations that need to be done for all m mixture components
            x = x.unsqueeze(1).expand(x.shape[0], self.num_components, x.shape[1])

        epch=0
        check=np.inf

        while(epch<=epochs )and ( check>=dl):
            old_likelihood=self.likelihood
            old_mean=self.mean
            old_var=self.var

            self.__em(x)
            self.likelihood=self.__likelihood(self.z,self.__gaus_prob(self.mean,self.var))

            if (self.likelihood==float("Inf")) or (self.likelihood==float("nan")):
                self.__init__(self.num_components,self.num_features)
                #reinitialization when likelihood has undefined values

            epch=epch +1
            check = self.likelihood - old_likelihood

            if check<0:
                self.__update_mean(old_mean)
                self.__update_var(old_var)
                 #revert to old parameters if likelihood decreases.

    def __gaus_prob(self,x,mean,var):
        """Calculating the likelihood of data given the latent variable z and the mean and variance
        of all of the individual gaussains in the mixture.
        Returns a tensor of dimension n x m x 1"""

        #Broadcasting tensors for parallel operations
        # 1 x m x f to  n x m x f
        mean = mean.expand(x.size(0), self.num_components, self.num_features)
        var = var.expand(x.size(0), self.num_components, self.num_features)

        # All of the following operations are element-wise operations
        distortion=tr.exp(-.5 * tr.sum((x - mean) * (x - mean) / var, 2, keepdim=True))
        normalization=tr.rsqrt(((2. * pi) ** self.num_features) * tr.prod(var, dim=2, keepdim=True) + self.eps)

        return normalization*distortion

    def __likelihood(self,z,gauss_prob):
        """Log-likelihood calculator for data given the model"""

        one_point_one_component=z * gauss_prob
        return tr.sum(tr.log(tr.sum(one_point_one_component,1) + self.eps))

    def __em(self,x):
        """One iteration of the expectation maximization
        the final update equations have been used rather than q function"""

        posterior= self.__e_step(self.z,self.__gaus_prob(x,self.mean,self.var))

        z,mean,var=self.__m_step(x,posterior)

        self.__update_z(z)
        self.__update_mean(mean)
        self.__update_var(var)

    def __e_step(self,z,gauss_prob):

        #Probability of genarating a point from a particular mixture component
        prob= z * gauss_prob
        return prob/(tr.sum(prob,1,keepdim=True) + self.eps)

    def __m_step(self,x,posterior):
        """Updates the model's parameters. This is the maximization step of the EM-algorithm."""
        # (n, m, 1) --> (1, m, 1)
        n_m = tr.sum(posterior, 0, keepdim=True)
        z_new = tr.div(n_m, tr.sum(n_m, 1, keepdim=True) + self.eps)
        # (n, m, f) --> (1, m, f)
        mean_new = tr.div(tr.sum(posterior * x, 0, keepdim=True), n_m + self.eps)
        # (n, m, f) --> (1, m, f)
        var_new = tr.div(tr.sum(posterior * (x - mean_new) * (x - mean_new), 0, keepdim=True), n_m + self.eps)

        return z_new, mean_new, var_new

    def predict(self,x,soft_align=False):
        """Assign input data x to a particular mixture component by evaluating likelihood under each.
        Soft alignment return a set of m probabilities instead"""

        if(len(x.shape)):
            x = x.unsqueeze(1).expand(x.size(0), self.num_components, x.size(1))

        gauss_prob=self.__gaus_prob(x,self.mean,self.var)

        if soft_align:
            return gauss_prob/(gauss_prob.sum(1,keepdim=True) + self.eps)
        else:
            _,predictions=tr.max(gauss_prob,1)
            return tr.squeeze(predictions).type(tr.LongStorage)



    def __update_mean(self, mean):
        assert mean.size() in [(self.num_components, self.num_features), (1, self.num_components,
                                                                        self.num_features)], "Unmatched Dimensions, expected: (%i, %i) or (1, %i, %i)  got: (%i)" % (self.num_components, self.num_features, self.num_components, self.num_features,str(mean.shape))
        if mean.size() == (self.num_components, self.num_features):
            self.mean = mean.unsqueeze(0)
        elif mean.size() == (1, self.num_components, self.num_features):
            self.mean.data = mean

    def __update_var(self, var):
        assert var.size() in [(self.num_components, self.num_features), (1, self.num_components,
                                                                         self.num_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
        self.num_components, self.num_features, self.num_components, self.num_features)

        if var.size() == (self.num_components, self.num_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.num_components, self.num_features):
            self.var.data = var

    def __update_z(self, z):

        assert z.size() in [
                (1, self.num_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (
            1, self.num_components, 1)

        self.z.data = z