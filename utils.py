import os
import math

import gnumpy as gnp
import numpy as np


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def sample_sigmoid_gnp(input):
    return gnp.rand(*input.shape) < gnp.logistic(input)


def sample_sigmoid_np(input):
    return np.random.rand(*input.shape) < gnp.logistic(input)


def reshape_and_tile_images(array, shape=(28, 28), n_cols=None):
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(math.ceil(float(array.shape[0])/n_cols))

    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            return array[ind].reshape(*shape, order='C')
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)


def gnp_repeat(array, n):
    '''gnp equivalent of np.repeat(array,n,axis=0) for a two-dimensional array'''
    return gnp.concatenate((array for i in xrange(n)), axis=1).reshape(array.shape[0]*n, array.shape[1])

def gnp_logmeanexp(array, axis = 0):
    '''works for axis<2 only'''
    assert axis<2, 'gnp_logmeanexp called with axis>=2'
    if axis == 0:
        m = gnp.max(array, axis = 0)[None,...]
        return gnp.log(gnp.mean(gnp.exp(array-m),axis=axis))+m
    elif axis == 1:
        m = gnp.max(array, axis = 1)[:,None,...]
        return gnp.log(gnp.mean(gnp.exp(array-m),axis=axis))[:,None,...]+m

def np_logmeanexp(array, axis, keepdims=False):
    m = np.max(array, axis = axis)
    logmeanexp = np.log(np.mean(np.exp(array-np.expand_dims(m,axis=axis)),axis=axis))+m
    if not keepdims:
        return logmeanexp
    else:
        return np.expand_dims(logmeanexp,axis=axis)


def weighted_choice(ps):
    '''ps is a 2d array with rows summing up to 1
    Generates random sample of range(0,ps.shape[1]) with weights in ps' rows'''
    n_samples, n_choices = ps.shape
    cumsum = np.cumsum(ps.reshape((-1,),order = 'C'))
    choices = np.random.rand(n_samples)+np.arange(n_samples)
    indices = np.searchsorted(cumsum,choices)
    return_indices = indices-np.arange(n_samples)*n_choices
    weird_indices = (return_indices>=n_choices)
    if np.sum(weird_indices)>0:
        print (ps)
    return_indices[weird_indices] = -1
    return return_indices
