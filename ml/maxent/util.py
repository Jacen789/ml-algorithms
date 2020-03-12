# -*- coding: utf-8 -*-

import math


def log_likelihood(classifier, gold):
    results = classifier.prob_classify_many([fs for (fs, l) in gold])
    ll = [pdist.prob(l) for ((fs, l), pdist) in zip(gold, results)]
    return math.log(sum(ll) / len(ll))


def accuracy(classifier, gold):
    results = classifier.classify_many([fs for (fs, l) in gold])
    correct = [l == r for ((fs, l), r) in zip(gold, results)]
    if correct:
        return sum(correct) / len(correct)
    else:
        return 0


class CutoffChecker(object):
    """
    A helper class that implements cutoff checks based on number of
    iterations and log likelihood.

    Accuracy cutoffs are also implemented, but they're almost never
    a good idea to use.
    """

    def __init__(self, cutoffs):
        self.cutoffs = cutoffs.copy()
        if 'min_ll' in cutoffs:
            cutoffs['min_ll'] = -abs(cutoffs['min_ll'])
        if 'min_lldelta' in cutoffs:
            cutoffs['min_lldelta'] = abs(cutoffs['min_lldelta'])
        self.ll = None
        self.acc = None
        self.iter = 1

    def check(self, classifier, train_toks):
        cutoffs = self.cutoffs
        self.iter += 1
        if 'max_iter' in cutoffs and self.iter >= cutoffs['max_iter']:
            return True  # iteration cutoff.

        new_ll = log_likelihood(classifier, train_toks)
        if math.isnan(new_ll):
            return True

        if 'min_ll' in cutoffs or 'min_lldelta' in cutoffs:
            if 'min_ll' in cutoffs and new_ll >= cutoffs['min_ll']:
                return True  # log likelihood cutoff
            if (
                'min_lldelta' in cutoffs
                and self.ll
                and ((new_ll - self.ll) <= abs(cutoffs['min_lldelta']))
            ):
                return True  # log likelihood delta cutoff
            self.ll = new_ll

        if 'max_acc' in cutoffs or 'min_accdelta' in cutoffs:
            new_acc = log_likelihood(classifier, train_toks)
            if 'max_acc' in cutoffs and new_acc >= cutoffs['max_acc']:
                return True  # log likelihood cutoff
            if (
                'min_accdelta' in cutoffs
                and self.acc
                and ((new_acc - self.acc) <= abs(cutoffs['min_accdelta']))
            ):
                return True  # log likelihood delta cutoff
            self.acc = new_acc

            return False  # no cutoff reached.
