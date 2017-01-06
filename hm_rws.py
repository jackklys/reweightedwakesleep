import numpy as np
import gnumpy as gnp
import utils
from itertools import izip


class Hm():
    def __init__(self, p_layers, q_layers, prior):
        '''
        p_net has all the generative layers starting from the one closest to the prior ending with the one closest to the data
        q_net has all the approximate inference layers starting from the one closest to the data
        '''
        self.p_layers = p_layers
        self.q_layers = q_layers
        self.prior = prior

    def q_samplesIx(self, x, repeat=1):
        sample = utils.gnp_repeat(x, repeat)
        yield sample
        for layer in self.q_layers:
            sample = layer.sampleIx(sample)
            yield sample

    def p_samplesIprior_sample(self, x, to_sample=True):
        sample = x
        yield sample
        for layer in self.p_layers[:-1]:
            sample = layer.sampleIx(sample)
            yield sample
        if to_sample:
            sample = self.p_layers[-1].sampleIx(sample)
            yield sample
        else:
            p = self.p_layers[-1].pIx(sample)
            yield p

    def log_pIq_samples(self, q_samples):
        logp = 0
        for layer, inpt, sample in izip(reversed(self.p_layers), q_samples[1:], q_samples):
            logp += layer.log_likelihood_sampleIx(sample, inpt)
        return logp

    def log_qIq_samples(self, q_samples):
        logq = 0
        for layer, inpt, sample in izip(self.q_layers, q_samples, q_samples[1:]):
            logq += layer.log_likelihood_sampleIx(sample, inpt)
        return logq

    def posterior_importance_weightsIq_samples(self, q_samples, repeat):
        log_q = self.log_qIq_samples(q_samples)
        log_p = self.log_pIq_samples(q_samples)
        log_prior_unnormalized = self.prior.log_likelihood_unnormalized(q_samples[-1])

        log_q = log_q.reshape(-1, repeat)
        log_p = log_p.reshape(-1, repeat)
        log_prior_unnormalized = log_prior_unnormalized.reshape(-1, repeat)

        log_weights = log_p + log_prior_unnormalized - log_q
        log_weights -= gnp.max(log_weights, axis=1)[:, None]
        weights = gnp.exp(log_weights)
        weights /= gnp.sum(weights, axis=1)[:, None]
        return weights

    def wake_sleep_reweight(self, minibatch, repeat, learning_rate, cd_steps):
        q_samples = list(self.q_samplesIx(minibatch, repeat))

        weights = self.posterior_importance_weightsIq_samples(q_samples, repeat=repeat)
        # weights /= weights.shape[0]
        weights = weights.reshape((-1, 1))
        # weights = np.empty((minibatch.shape[0]*repeat,1))
        # weights.fill(1./minibatch.shape[0]*repeat)
        # weights = gnp.garray(weights)

        self.update_q_wake(learning_rate=learning_rate, q_samples=q_samples, weights=weights)
        self.prior.increase_weighted_log_likelihood(learning_rate=learning_rate, samples=q_samples[-1], weights=weights, cd_steps=cd_steps)
        self.update_p_wake(learning_rate=learning_rate, q_samples=q_samples, weights=weights)

    def update_q_wake(self, learning_rate, q_samples, weights):
        for layer, inpt, sample in izip(self.q_layers, q_samples, q_samples[1:]):
            layer.increase_weighted_log_likelihood(samples=sample, inputs=inpt, weights=weights, learning_rate=learning_rate)

    def update_p_wake(self, learning_rate, q_samples, weights):
        for layer, inpt, sample in izip(reversed(self.p_layers), q_samples[1:], q_samples):
            layer.increase_weighted_log_likelihood(samples=sample, inputs=inpt, weights=weights, learning_rate=learning_rate)


if __name__ == '__main__':
    import argparse
    import os
    import config
    import prob_distributions
    import nn
    import rbm
    import evaluate
    import cPickle as pkl

    import datasets
    dataset = datasets.mnist()

    parser = argparse.ArgumentParser(description='Train a DBN with jont wake sleep training procedure on the MNIST dataset')
    parser.add_argument('--learning_rate1', '-l1', type=float, default=3e-3)
    parser.add_argument('--learning_rate2', '-l2', type=float, default=3e-6)
    parser.add_argument('--learning_ratep', '-lp', type=float, default=1e-3)
    parser.add_argument('--cd_steps', '-c', type=int, default=10)
    parser.add_argument('--num_epochs', '-e', type=int, default=50)
    parser.add_argument('--num_pre_epochs', '-p', type=int, default=20)
    parser.add_argument('--repeat', '-k', type=int, default=10)
    parser.add_argument('--units', '-u', nargs='+', type=int, default=[500,500]) #usage: -u 400 200 will make [400, 200] first being bottom
    args = parser.parse_args()

    num_epochs = args.num_epochs
    num_pre_epochs = args.num_pre_epochs
    learning_rate1 = args.learning_rate1
    learning_rate2 = args.learning_rate2
    learning_ratep = args.learning_ratep
    cd_steps = args.cd_steps
    repeat = args.repeat
    u = args.units
    minibatch_size = 20
    data_dim = dataset.get_data_dim()

    directory_name = os.path.join(config.RESULTS_DIR, 'hm_rws{}'.format(u))
    directory_name = os.path.join(directory_name, 'e{}c{}p{}k{}l1{}l2{}lp{}'.format(num_epochs, cd_steps, num_pre_epochs, repeat, learning_rate1, learning_rate2, learning_ratep))
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    u.insert(0,data_dim)
    minibatches_per_epoch = dataset.get_n_examples('train') // minibatch_size
    model = Hm([], [], prior=0)

    '''pretraining layers'''
    for u_prev, u_next in izip(u[:-2],u[1:-1]):
        layer_pretrain = rbm.Rbm.random(u_prev,u_next)
        for i in xrange(num_pre_epochs):
            for j in xrange(minibatches_per_epoch):
                minibatch = dataset.get_minibatch_at_index(j, minibatch_size=minibatch_size)
                samples = list(model.q_samplesIx(minibatch))
                layer_pretrain.increase_weighted_log_likelihood(samples=samples[-1], weights=1./minibatch_size, learning_rate=learning_ratep, cd_steps=cd_steps)
        model.p_layers.insert(0,prob_distributions.Bernoulli(nn.Linear(layer_pretrain.w.T,layer_pretrain.bv)))
        model.q_layers.append(prob_distributions.Bernoulli(nn.Linear(layer_pretrain.w,layer_pretrain.bh)))

    '''pretraining prior'''
    prior_pretrain = rbm.Rbm.random(u[-2], u[-1])
    for i in xrange(num_pre_epochs):
        for j in xrange(minibatches_per_epoch):
            minibatch = dataset.get_minibatch_at_index(j, minibatch_size=minibatch_size)
            samples = list(model.q_samplesIx(minibatch))
            prior_pretrain.increase_weighted_log_likelihood(samples=samples[-1], weights=1./minibatch_size, learning_rate=learning_ratep, cd_steps=cd_steps)
    model.prior = prior_pretrain

    '''wake-sleep'''
    if num_epochs is not 0: delta = (learning_rate1-learning_rate2)/(minibatches_per_epoch*num_epochs)
    for i in xrange(num_epochs):
        for j in xrange(minibatches_per_epoch):
            learning_rate = learning_rate1 - (i*minibatches_per_epoch+j)*delta
            minibatch = dataset.get_minibatch_at_index(j, minibatch_size=minibatch_size)
            model.wake_sleep_reweight(minibatch=minibatch, repeat=repeat, learning_rate=learning_rate, cd_steps=cd_steps)

    f = open(os.path.join(directory_name,'hm_rws.pkl'), 'wb')
    pkl.dump(model,f,protocl=pkl.HIGHEST_PROTOCOL)
    f.close()

    evaluate.visualize_samples(directory_name, model)
    evaluate.ais(directory_name, dataset, model)
