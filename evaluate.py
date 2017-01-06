import numpy as np
import gnumpy as gnp
import utils
from itertools import izip
import os
import config


def visualize_samples(directory_name, model, steps=10000, runs=50):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imshow(utils.reshape_and_tile_images(model.p_layers[-1].logit_network.w.asarray()), cmap='Greys')
    plt.savefig(os.path.join(directory_name, 'p_weights.jpg'))
    plt.imshow(utils.reshape_and_tile_images(model.q_layers[0].logit_network.w.asarray().T), cmap='Greys')
    plt.savefig(os.path.join(directory_name, 'q_weights.jpg'))
    prior_sample = model.prior.run_mcmc_chain(steps, num_samples=runs, to_sample=True)
    samples = list(model.p_samplesIprior_sample(prior_sample, to_sample=False))
    plt.imshow(utils.reshape_and_tile_images(samples[-1].asarray()), cmap='Greys')
    plt.savefig(os.path.join(directory_name, 'samples.jpg'))

def ais(directory_name, dataset, model, steps=100000, runs=15000, bv_0=None):
    if bv_0 is None:
        bv_0 = gnp.zeros((1,model.prior.bv.shape[1]))
    prior_log_partition_function_uni = model.prior.log_partition_function(runs,steps,bv_0)
    # bv_0 = dataset.get_train_logit_of_mean()
    # prior_log_partition_function_br = model.prior.log_partition_function(100,150,bv_0)
    log_p = 0
    minibatch = dataset.get_minibatch_at_index(0, minibatch_size=dataset.get_n_examples(subdataset='test'), subdataset='test')
    q_samples = list(model.q_samplesIx(minibatch))
    log_p += model.log_pIq_samples(q_samples) - model.log_qIq_samples(q_samples)
    log_p += model.prior.log_likelihood_unnormalized(q_samples[-1])
    x = gnp.sum(log_p,axis=0)/dataset.get_n_examples('test')
    AIS_log_likelihood_uni = x - prior_log_partition_function_uni
    with open(os.path.join(config.RESULTS_DIR, 'results.txt'), 'a') as g:
        g.write(directory_name[22:] + "\n" + str(prior_log_partition_function_uni) + "\n" + str(AIS_log_likelihood_uni) + "\n\n")

