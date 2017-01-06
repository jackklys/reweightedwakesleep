import gnumpy as gnp
import numpy as np
import utils


class Linear:
    def __init__(self, w, b):
        self.w = gnp.garray(w)
        self.b = gnp.garray(b)

    def yIx(self, x):
        return x.dot(self.w) + self.b

    def grad_input(self, grad_output):
        return grad_output.dot(self.w.T)

    def update_paramsIgrad(self, grad_output, inputs, learning_rate):
        self.w += (inputs.T.dot(grad_output))*learning_rate
        self.b += (gnp.sum(grad_output, axis=0)[None, :])*learning_rate

    @staticmethod
    def random(n_in, n_out, factor=1., seed=123):
        '''A randomly initialized linear layer.
        When factor is 1, the initialization is uniform as in Glorot, Bengio, 2010,
        assuming the layer is intended to be followed by the tanh nonlinearity.'''
        random_state = np.random.RandomState(seed)

        scale = factor*np.sqrt(6./(n_in+n_out))
        return Linear(w=gnp.garray(random_state.uniform(low=-scale, high=scale, size=(n_in, n_out))),
                      b=gnp.garray(np.zeros((1, n_out))))

class Chain:
    def __init__(self, layers):
        self.layers = layers

    def yIx(self, x):
        for output in self.all_yIx(x):
            pass
        return output

    def all_yIx(self, x):
        output = x
        yield output
        for layer in self.layers:
            output = layer.yIx(output)
            yield output

    def update_paramsIgrad(self, inputs, grad_output, learning_rate):
        outputs = list(self.all_yIx(inputs))
        for layer, inpt in zip(self.layers, outputs):
            layer.update_params(inputs=inpt, grad_output=grad_output, learning_rate=learning_rate)
            grad_output = layer.grad_input(grad_output=grad_output, learning_rate=learning_rate)
