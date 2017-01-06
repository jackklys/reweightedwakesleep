import gnumpy as gnp
import numpy as np
import utils


def bernoulli_log_likelihoodIlogit(x, logit):
    '''
    log p(x) for x a binary vector, and p(1):p(0)=e^logit : 1
    '''
    # return gnp.sum((x-(logit > 0))*logit-gnp.log(1.+gnp.exp(-gnp.abs(logit))), axis=1)
    x_signs = 2*x - 1
    return -gnp.sum(gnp.log_1_plus_exp(-x_signs*logit),axis=1)


class Bernoulli:
    def __init__(self, logit_network):
        self.logit_network = logit_network

    def logitIx(self, x):
        return self.logit_network.yIx(x)

    def pIlogit(self, logit):
        return gnp.logistic(logit)

    def pIx(self, x):
        return self.pIlogit(self.logitIx(x))

    def sampleIx(self, x):
        p = self.pIx(x)
        return gnp.rand(*p.shape) < p

    def log_likelihood_sampleIx(self, sample, x):
        return bernoulli_log_likelihoodIlogit(sample, self.logitIx(x))

    def increase_weighted_log_likelihood(self, samples, inputs, weights, learning_rate):
        p = self.pIx(inputs)
        grad = weights * (samples - p)
        self.logit_network.update_paramsIgrad(inputs=inputs, grad_output=grad, learning_rate=learning_rate)
