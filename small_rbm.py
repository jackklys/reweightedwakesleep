import utils
import numpy as np
import gnumpy as gnp

import os
import argparse
import config


class Rbm():
    def __init__(self, w, bv, bh):
        self.w = gnp.garray(w)
        self.bv = gnp.garray(bv.reshape((1, -1)))
        self.bh = gnp.garray(bh.reshape((1, -1)))

    def p_hIv(self, v):
        return gnp.logistic(v.dot(self.w)+self.bh)

    def run_mcmc_chain(self, num_steps, v=None, num_samples=None, to_sample=False):
        if v is None:
            v = gnp.rand(num_samples, self.bv.shape[1]) < 0.5  # initialize at uniform random sample

        for _ in xrange(num_steps-1):
            h = utils.sample_sigmoid_gnp(v.dot(self.w)+self.bh)
            v = utils.sample_sigmoid_gnp(h.dot(self.w.T)+self.bv)

        h = utils.sample_sigmoid_gnp(v.dot(self.w)+self.bh)
        if to_sample:
            v = utils.sample_sigmoid_gnp(h.dot(self.w.T)+self.bv)
        else:
            v = gnp.logistic(h.dot(self.w.T)+self.bv)

        return v

    def log_likelihood_unnormalized(self, v):
        return log_p_v_unnormalized(v, self.w, self.bv, self.bh)

    @staticmethod
    def random(n_vis, n_hid, bias_v=None, bias_h=None, factor=1.):
        w = 2.*(gnp.rand(n_vis, n_hid)-0.5)/gnp.sqrt(n_vis+n_hid)
        bv = bias_v if bias_v is not None else gnp.zeros((1, n_vis))
        bh = bias_h if bias_h is not None else gnp.zeros((1, n_hid))
        return Rbm(w, bv, bh)

    def increase_weighted_log_likelihood(self, samples, weights, cd_steps, learning_rate):
        v = samples
        p_h_0 = gnp.logistic(v.dot(self.w)+self.bh)
        for _ in xrange(cd_steps):
            h = utils.sample_sigmoid_gnp(v.dot(self.w)+self.bh)
            v = utils.sample_sigmoid_gnp(h.dot(self.w.T)+self.bv)
        p_h = gnp.logistic(v.dot(self.w)+self.bh)

        wv_0 = weights * samples
        wv = weights * v
        grad_w = wv_0.T.dot(p_h_0) - wv.T.dot(p_h)
        grad_bv = (gnp.sum(wv_0, axis=0)-gnp.sum(wv, axis=0))[None, :]
        grad_bh = (gnp.sum(weights*p_h_0, axis=0)-gnp.sum(weights*p_h, axis=0))[None, :]
        self.w += learning_rate*grad_w
        self.bv += learning_rate*grad_bv
        self.bh += learning_rate*grad_bh

    def reconstruction_error(self, x, repeats=5):
        v = utils.gnp_repeat(x, repeats)
        h = utils.sample_sigmoid_gnp(v.dot(self.w)+self.bh)
        pv = gnp.logistic(h.dot(self.w.T)+self.bv)
        return gnp.sqrt(gnp.mean((v-pv)**2))

    def parameters(self,bv_0,alpha):
        return self.w*alpha, self.bv*alpha+bv_0*(1-alpha), self.bh*alpha

    def log_partition_function(self,numpasses,depth,bv_0):
        v = gnp.randn(numpasses,self.bv.shape[1])
        v = v < 0.5
        v_0 = v
        a = gnp.zeros((numpasses,1))
        for k in range(depth*2):
            w, bv, bh = self.parameters(bv_0,(k+1)/(2.0*depth))
            if k % 2 == 0:
                h = utils.sample_sigmoid_gnp(v.dot(w)+bh)
                v_signs = 2*v - 1
                log_p_vIh = -gnp.sum(gnp.log_1_plus_exp(-v_signs*(gnp.dot(h,w.T)+bv)),axis=1)[:,None]
                h_signs = 2*h - 1
                log_p_hIv = -gnp.sum(gnp.log_1_plus_exp(-h_signs*(gnp.dot(v,w)+bh)),axis=1)[:,None]
                da = log_p_vIh - log_p_hIv
            else:
                v = utils.sample_sigmoid_gnp(h.dot(w.T)+bv)
                h_signs = 2*h - 1
                log_p_hIv = -gnp.sum(gnp.log_1_plus_exp(-h_signs*(gnp.dot(v,w)+bh)),axis=1)[:,None]
                v_signs = 2*v - 1
                log_p_vIh = -gnp.sum(gnp.log_1_plus_exp(-v_signs*(gnp.dot(h,w.T)+bv)),axis=1)[:,None]
                da = log_p_hIv - log_p_vIh
            a += da
        a += (self.log_likelihood_unnormalized(v))[:,None]
        # a += gnp.garray([(self.bv.shape[1])*(gnp.log(2))])[:,None]
        a += gnp.sum(gnp.log_1_plus_exp(bv_0),axis=1)[:,None] - gnp.sum(v_0*bv_0,axis=1)[:,None]
        b = gnp.max(a,axis=0)[:,None]
        a = gnp.exp(a - b)
        return gnp.log(gnp.sum(a,axis=0)/a.shape[0])+b

    def exact_log_partition_function(self):
        h = np.array([[0],[1]])
        for i in range(self.bh.shape[1]-1):
            h1 = np.concatenate((np.zeros((h.shape[0],1)),np.ones((h.shape[0],1))),axis=0)
            h = np.concatenate((h,h),axis=0)
            h = np.concatenate((h1,h),axis=1)
        h1 = np.split(h,128,axis=0)
        a= gnp.zeros(128)
        i=0
        for j in h1:
            log_p_h_unnormalized = gnp.sum(gnp.log_1_plus_exp(gnp.dot(gnp.garray(j),(self.w).T)+self.bv),axis=1) \
                 + gnp.sum(gnp.garray(j)*self.bh, axis=1)
            max = gnp.max(log_p_h_unnormalized,axis=0)
            b = gnp.exp(log_p_h_unnormalized-max)
            a[i] = gnp.log(gnp.sum(b,axis=0)) + max
            i+=1
        max = gnp.max(a,axis=0)
        b = gnp.exp(a-max)
        return gnp.log(gnp.sum(b,axis=0)) + max


def log_p_v_unnormalized(v, w, bv, bh):
        return gnp.sum(gnp.log_1_plus_exp(gnp.dot(v, w)+bh),axis=1) + gnp.sum(v*bv, axis=1)


# def reconstruction_error_and_sparsity_drpt_gnp(v,W,bv,bh,repeats=5):
#     v = utils.gnp_repeat(v,repeats)
#     h = utils.sample_sigmoid_gnp(v.dot(W)+bh)
#     mask = gnp.rand(*h.shape) < 0.5
#     pv = gnp.logistic((h*mask).dot(W.T)+bv)
#     return gnp.mean(gnp.abs(v-pv)), gnp.mean(h)

# def run_annealed_mcmc_chain(rbm, num_steps, v=None, num_samples=None):
    
#     if v is None:
#         v = 1.*(np.random.rand(num_samples,rbm.bv.shape[1])<0.5)
#     ans = []
#     for i in xrange(num_steps-1):
#         beta = float(i)/num_steps
#         h = utils.sample_sigmoid_np(beta*(v.dot(rbm.w)+rbm.bh))
#         v_probs = scipy.special.expit(beta*(h.dot(rbm.w.T)+rbm.bv))
#         ans.append(v_probs)
#         v = 1.*(np.random.rand(*v_probs.shape)<v_probs)

#     h = utils.sample_sigmoid_np(v.dot(rbm.w)+rbm.bh)
#     v = scipy.special.expit(h.dot(rbm.w.T)+rbm.bv)
#     ans.append(v)
#     return ans

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an RBM with CD training procedure on the MNIST dataset')
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-3)
    parser.add_argument('--cd_steps', '-c', type=int, default=1)
    parser.add_argument('--num_epochs', '-e', type=int, default=10)
    args = parser.parse_args()

    num_epochs = args.num_epochs
    minibatch_size = 20
    learning_rate = args.learning_rate
    cd_steps = args.cd_steps

    directory_name = os.path.join(config.RESULTS_DIR, 'small_rbm')
    directory_name = os.path.join(directory_name, 'l{}c{}e{}'.format(learning_rate, cd_steps, num_epochs))
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    import datasets
    dataset = datasets.mnist()

    model = Rbm.random(dataset.get_data_dim(), 25, bias_v=dataset.get_train_logit_of_mean())

    minibatches_per_epoch = dataset.get_n_examples('train') // minibatch_size
    for i in xrange(num_epochs):
        for j in xrange(minibatches_per_epoch):
            minibatch = dataset.get_minibatch_at_index(j, minibatch_size=minibatch_size)
            model.increase_weighted_log_likelihood(samples=minibatch, weights=1./minibatch_size, learning_rate=learning_rate, cd_steps=cd_steps)

    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # plt.imshow(utils.reshape_and_tile_images(model.w.asarray().T), cmap='Greys')
    # plt.savefig(os.path.join(directory_name, 'weights.jpg'))
    # reconstruction_error = model.reconstruction_error(dataset.get_minibatch_at_index(0, 100))
    # with open(os.path.join(directory_name, 'reconstruction_error.txt'), 'w') as f:
    #     f.write(str(reconstruction_error))
    # samples = model.run_mcmc_chain(100000, num_samples=50)
    # plt.imshow(utils.reshape_and_tile_images(samples.asarray()), cmap='Greys')
    # plt.savefig(os.path.join(directory_name, 'samples.jpg'))

    exact_log_partition_function = model.exact_log_partition_function()
    bv_0=gnp.zeros((1,dataset.get_data_dim()))
    log_partition_function_uni = model.log_partition_function(10,1000,bv_0)
    bv_0 = dataset.get_train_logit_of_mean()
    log_partition_function_br1 = model.log_partition_function(10,1000,bv_0)
    count = dataset.get_train_mean()*dataset.get_n_examples()
    lp = 5*minibatches_per_epoch
    bv_0 = (count + lp) / (dataset.get_n_examples() + lp)
    bv_0 = gnp.log(bv_0) - gnp.log(1 - bv_0)
    log_partition_function_br2 = model.log_partition_function(10,1000,bv_0)
    minibatches_per_epoch = dataset.get_n_examples('test') // minibatch_size
    log_p = 0
    for i in xrange(minibatches_per_epoch):
        minibatch = dataset.get_minibatch_at_index(i, minibatch_size=minibatch_size, subdataset='test')
        log_p += model.log_likelihood_unnormalized(minibatch)
    x = gnp.sum(log_p,axis=0)/dataset.get_n_examples('test')
    AIS_log_likelihood_uni =  x - log_partition_function_uni
    AIS_log_likelihood_br1 = x - log_partition_function_br1
    AIS_log_likelihood_br2 = x - log_partition_function_br2
    exact_log_likelihood = x - exact_log_partition_function
    with open(os.path.join(directory_name, 'AIS_loglikelihood_estimate.txt'), 'w') as f:
        f.write(str(exact_log_partition_function) + "\n" + str(exact_log_likelihood) + "\n\n")
        f.write(str(log_partition_function_uni) + "\n" + str(AIS_log_likelihood_uni) + "\n\n")
        f.write(str(log_partition_function_br1) + "\n" + str(AIS_log_likelihood_br1) + "\n\n")
        f.write(str(log_partition_function_br2) + "\n" + str(AIS_log_likelihood_br2) + "\n\n")
