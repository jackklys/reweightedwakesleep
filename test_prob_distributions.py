import unittest

import gnumpy as gnp
from prob_distributions import bernoulli_log_likelihoodIlogit


class TestBernoulli(unittest.TestCase):
    def test_probs_sum_to_one(self):
        logit = gnp.randn(1, 2)
        all_outcomes = gnp.garray([[0., 0.],
                                   [1., 0.],
                                   [0., 1.],
                                   [1., 1.]])
        log_probs = bernoulli_log_likelihoodIlogit(all_outcomes, logit)
        self.assertAlmostEqual(gnp.sum(gnp.exp(log_probs)), 1.)

    def test_zero_logit_prob_one_half(self):
        logit = gnp.garray([[0.]])
        log_probs = bernoulli_log_likelihoodIlogit(gnp.garray([[0.], [1.]]), logit)
        self.assertAlmostEqual(gnp.sum(gnp.abs(gnp.exp(log_probs)-0.5)), 0.)


if __name__ == '__main__':
    unittest.main()
