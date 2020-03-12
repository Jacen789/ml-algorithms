# -*- coding: utf-8 -*-

import math
import random
import warnings
from functools import reduce

_NINF = float('-1e300')


class DictionaryProbDist(object):
    """
    A probability distribution whose probabilities are directly
    specified by a given dictionary.  The given dictionary maps
    samples to probabilities.
    """

    SUM_TO_ONE = True
    """True if the probabilities of the samples in this probability
       distribution will always sum to one."""

    def __init__(self, prob_dict=None, log=False, normalize=False):
        """
        Construct a new probability distribution from the given
        dictionary, which maps values to probabilities (or to log
        probabilities, if ``log`` is true).  If ``normalize`` is
        true, then the probability values are scaled by a constant
        factor such that they sum to 1.

        If called without arguments, the resulting probability
        distribution assigns zero probability to all values.
        """

        self._prob_dict = prob_dict.copy() if prob_dict is not None else {}
        self._log = log

        # Normalize the distribution, if requested.
        if normalize:
            if len(prob_dict) == 0:
                raise ValueError(
                    'A DictionaryProbDist must have at least one sample '
                    + 'before it can be normalized.'
                )
            if log:
                value_sum = sum_logs(list(self._prob_dict.values()))
                if value_sum <= _NINF:
                    logp = math.log(1.0 / len(prob_dict), 2)
                    for x in prob_dict:
                        self._prob_dict[x] = logp
                else:
                    for (x, p) in self._prob_dict.items():
                        self._prob_dict[x] -= value_sum
            else:
                value_sum = sum(self._prob_dict.values())
                if value_sum == 0:
                    p = 1.0 / len(prob_dict)
                    for x in prob_dict:
                        self._prob_dict[x] = p
                else:
                    norm_factor = 1.0 / value_sum
                    for (x, p) in self._prob_dict.items():
                        self._prob_dict[x] *= norm_factor

    def prob(self, sample):
        if self._log:
            return 2 ** (self._prob_dict[sample]) if sample in self._prob_dict else 0
        else:
            return self._prob_dict.get(sample, 0)

    def logprob(self, sample):
        if self._log:
            return self._prob_dict.get(sample, _NINF)
        else:
            if sample not in self._prob_dict:
                return _NINF
            elif self._prob_dict[sample] == 0:
                return _NINF
            else:
                return math.log(self._prob_dict[sample], 2)

    def max(self):
        if not hasattr(self, '_max'):
            self._max = max((p, v) for (v, p) in self._prob_dict.items())[1]
        return self._max

    def samples(self):
        return self._prob_dict.keys()

    def __repr__(self):
        return '<ProbDist with %d samples>' % len(self._prob_dict)

    # cf self.SUM_TO_ONE
    def discount(self):
        """
        Return the ratio by which counts are discounted on average: c*/c

        :rtype: float
        """
        return 0.0

    # Subclasses should define more efficient implementations of this,
    # where possible.
    def generate(self):
        """
        Return a randomly selected sample from this probability distribution.
        The probability of returning each sample ``samp`` is equal to
        ``self.prob(samp)``.
        """
        p = random.random()
        p_init = p
        for sample in self.samples():
            p -= self.prob(sample)
            if p <= 0:
                return sample
        # allow for some rounding error:
        if p < 0.0001:
            return sample
        # we *should* never get here
        if self.SUM_TO_ONE:
            warnings.warn(
                "Probability distribution %r sums to %r; generate()"
                " is returning an arbitrary sample." % (self, p_init - p)
            )
        return random.choice(list(self.samples()))


##//////////////////////////////////////////////////////
## Adding in log-space.
##//////////////////////////////////////////////////////

# If the difference is bigger than this, then just take the bigger one:
_ADD_LOGS_MAX_DIFF = math.log(1e-30, 2)


def add_logs(logx, logy):
    """
    Given two numbers ``logx`` = *log(x)* and ``logy`` = *log(y)*, return
    *log(x+y)*.  Conceptually, this is the same as returning
    ``log(2**(logx)+2**(logy))``, but the actual implementation
    avoids overflow errors that could result from direct computation.
    """
    if logx < logy + _ADD_LOGS_MAX_DIFF:
        return logy
    if logy < logx + _ADD_LOGS_MAX_DIFF:
        return logx
    base = min(logx, logy)
    return base + math.log(2 ** (logx - base) + 2 ** (logy - base), 2)


def sum_logs(logs):
    return reduce(add_logs, logs[1:], logs[0]) if len(logs) != 0 else _NINF
