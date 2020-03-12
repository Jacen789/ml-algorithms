# -*- coding: utf-8 -*-

import numpy as np

from six import integer_types
from collections import defaultdict

from maxent.util import CutoffChecker, accuracy, log_likelihood
from maxent.probability import DictionaryProbDist


class MaxentClassifier(object):

    def __init__(self, encoding, weights, logarithmic=True):
        self._encoding = encoding
        self._weights = weights
        self._logarithmic = logarithmic
        assert encoding.length() == len(weights)

    def labels(self):
        return self._encoding.labels()

    def set_weights(self, new_weights):
        self._weights = new_weights
        assert self._encoding.length() == len(new_weights)

    def weights(self):
        return self._weights

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        prob_dict = {}
        for label in self._encoding.labels():
            feature_vector = self._encoding.encode(featureset, label)

            if self._logarithmic:
                total = 0.0
                for (f_id, f_val) in feature_vector:
                    total += self._weights[f_id] * f_val
                prob_dict[label] = total
            else:
                prod = 1.0
                for (f_id, f_val) in feature_vector:
                    prod *= self._weights[f_id] ** f_val
                prob_dict[label] = prod

        return DictionaryProbDist(prob_dict, log=self._logarithmic, normalize=True)

    def classify_many(self, featuresets):
        return [self.classify(fs) for fs in featuresets]

    def prob_classify_many(self, featuresets):
        return [self.prob_classify(fs) for fs in featuresets]

    def explain(self, featureset, columns=4):
        descr_width = 50
        TEMPLATE = '  %-' + str(descr_width - 2) + 's%s%8.3f'

        pdist = self.prob_classify(featureset)
        labels = sorted(pdist.samples(), key=pdist.prob, reverse=True)
        labels = labels[:columns]
        print('  Feature'.ljust(descr_width) + ''.join('%8s' % (("%s" % l)[:7]) for l in labels))
        print('  ' + '-' * (descr_width - 2 + 8 * len(labels)))
        sums = defaultdict(int)
        for i, label in enumerate(labels):
            feature_vector = self._encoding.encode(featureset, label)
            feature_vector.sort(key=lambda fid__: abs(self._weights[fid__[0]]), reverse=True)
            for (f_id, f_val) in feature_vector:
                if self._logarithmic:
                    score = self._weights[f_id] * f_val
                else:
                    score = self._weights[f_id] ** f_val
                descr = self._encoding.describe(f_id)
                descr = descr.split(' and label is ')[0]
                descr += ' (%s)' % f_val
                if len(descr) > 47:
                    descr = descr[:44] + '...'
                print(TEMPLATE % (descr, i * 8 * ' ', score))
                sums[label] += score
        print('  ' + '-' * (descr_width - 1 + 8 * len(labels)))
        print('  TOTAL:'.ljust(descr_width) + ''.join('%8.3f' % sums[l] for l in labels))
        print('  PROBS:'.ljust(descr_width) + ''.join('%8.3f' % pdist.prob(l) for l in labels))

    def most_informative_features(self, n=10):
        if hasattr(self, '_most_informative_features'):
            return self._most_informative_features[:n]
        else:
            self._most_informative_features = sorted(
                list(range(len(self._weights))),
                key=lambda fid: abs(self._weights[fid]),
                reverse=True,
            )
            return self._most_informative_features[:n]

    def show_most_informative_features(self, n=10, show='all'):
        fids = self.most_informative_features(None)
        if show == 'pos':
            fids = [fid for fid in fids if self._weights[fid] > 0]
        elif show == 'neg':
            fids = [fid for fid in fids if self._weights[fid] < 0]
        for fid in fids[:n]:
            print('%8.3f %s' % (self._weights[fid], self._encoding.describe(fid)))

    def __repr__(self):
        return '<ConditionalExponentialClassifier: %d labels, %d features>' % (
            len(self._encoding.labels()),
            self._encoding.length(),
        )

    ALGORITHMS = ['GIS', 'IIS']

    @classmethod
    def train(cls, train_toks, validation_toks=None, algorithm=None, trace=3, encoding=None, labels=None, **cutoffs):
        """
        Train a new maxent classifier based on the given corpus of
        training samples.  This classifier will have its weights
        chosen to maximize entropy while remaining empirically
        consistent with the training corpus.

        :rtype: MaxentClassifier
        :return: The new maxent classifier

        :type train_toks: list
        :param train_toks: Training data, represented as a list of
            pairs, the first member of which is a featureset,
            and the second of which is a classification label.

        :type validation_toks: list
        :param validation_toks: Validation data, represented as a list of
            pairs, the first member of which is a featureset,
            and the second of which is a classification label.

        :type algorithm: str
        :param algorithm: A case-insensitive string, specifying which
            algorithm should be used to train the classifier.  The
            following algorithms are currently available.

            - Iterative Scaling Methods: Generalized Iterative Scaling (``'GIS'``),
              Improved Iterative Scaling (``'IIS'``)

            The default algorithm is ``'IIS'``.

        :type trace: int
        :param trace: The level of diagnostic tracing output to produce.
            Higher values produce more verbose output.
        :type encoding: BinaryMaxentFeatureEncoding
        :param encoding: A feature encoding, used to convert featuresets
            into feature vectors.  If none is specified, then a
            ``BinaryMaxentFeatureEncoding`` will be built based on the
            features that are attested in the training corpus.
        :type labels: list(str)
        :param labels: The set of possible labels.  If none is given, then
            the set of all labels attested in the training data will be
            used instead.
        :param cutoffs: Arguments specifying various conditions under
            which the training should be halted.

            - ``max_iter=v``: Terminate after ``v`` iterations.
            - ``min_ll=v``: Terminate after the negative average
              log-likelihood drops under ``v``.
            - ``min_lldelta=v``: Terminate if a single iteration improves
              log likelihood by less than ``v``.
        """
        if algorithm is None:
            algorithm = 'iis'
        for key in cutoffs:
            if key not in ('max_iter', 'min_ll', 'min_lldelta', 'max_acc', 'min_accdelta'):
                raise TypeError('Unexpected keyword arg %r' % key)
        algorithm = algorithm.lower()
        if algorithm == 'iis':
            return train_maxent_classifier_with_iis(train_toks, validation_toks, trace, encoding, labels, **cutoffs)
        elif algorithm == 'gis':
            return train_maxent_classifier_with_gis(train_toks, validation_toks, trace, encoding, labels, **cutoffs)
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)


ConditionalExponentialClassifier = MaxentClassifier


class BinaryMaxentFeatureEncoding(object):

    def __init__(self, labels, mapping, unseen_features=False, alwayson_features=False):
        if set(mapping.values()) != set(range(len(mapping))):
            raise ValueError('Mapping values must be exactly the set of integers from 0...len(mapping)')

        self._labels = list(labels)
        self._mapping = mapping
        self._length = len(mapping)
        self._alwayson = None
        self._unseen = None

        if alwayson_features:
            self._alwayson = dict((label, i + self._length) for (i, label) in enumerate(labels))
            self._length += len(self._alwayson)

        if unseen_features:
            fnames = set(fname for (fname, fval, label) in mapping)
            self._unseen = dict((fname, i + self._length) for (i, fname) in enumerate(fnames))
            self._length += len(fnames)

    def encode(self, featureset, label):
        encoding = []

        for fname, fval in featureset.items():
            if (fname, fval, label) in self._mapping:
                encoding.append((self._mapping[fname, fval, label], 1))
            elif self._unseen:
                for label2 in self._labels:
                    if (fname, fval, label2) in self._mapping:
                        break
                else:
                    if fname in self._unseen:
                        encoding.append((self._unseen[fname], 1))

        if self._alwayson and label in self._alwayson:
            encoding.append((self._alwayson[label], 1))

        return encoding

    def describe(self, f_id):
        if not isinstance(f_id, integer_types):
            raise TypeError('describe() expected an int')
        try:
            self._inv_mapping
        except AttributeError:
            self._inv_mapping = [-1] * len(self._mapping)
            for (info, i) in self._mapping.items():
                self._inv_mapping[i] = info

        if f_id < len(self._mapping):
            (fname, fval, label) = self._inv_mapping[f_id]
            return '%s==%r and label is %r' % (fname, fval, label)
        elif self._alwayson and f_id in self._alwayson.values():
            for (label, f_id2) in self._alwayson.items():
                if f_id == f_id2:
                    return 'label is %r' % label
        elif self._unseen and f_id in self._unseen.values():
            for (fname, f_id2) in self._unseen.items():
                if f_id == f_id2:
                    return '%s is unseen' % fname
        else:
            raise ValueError('Bad feature id')

    def labels(self):
        return self._labels

    def length(self):
        return self._length

    @classmethod
    def train(cls, train_toks, count_cutoff=0, labels=None, **options):
        """
        Construct and return new feature encoding, based on a given
        training corpus ``train_toks``.

        :type train_toks: list(tuple(dict, str))
        :param train_toks: Training data, represented as a list of
            pairs, the first member of which is a feature dictionary,
            and the second of which is a classification label.

        :type count_cutoff: int
        :param count_cutoff: A cutoff value that is used to discard
            rare joint-features.  If a joint-feature's value is 1
            fewer than ``count_cutoff`` times in the training corpus,
            then that joint-feature is not included in the generated
            encoding.

        :type labels: list
        :param labels: A list of labels that should be used by the
            classifier.  If not specified, then the set of labels
            attested in ``train_toks`` will be used.

        :param options: Extra parameters for the constructor, such as
            ``unseen_features`` and ``alwayson_features``.
        """
        mapping = {}
        seen_labels = set()
        count = defaultdict(int)

        for (tok, label) in train_toks:
            if labels and label not in labels:
                raise ValueError('Unexpected label %s' % label)
            seen_labels.add(label)

            for (fname, fval) in tok.items():
                count[fname, fval] += 1
                if count[fname, fval] >= count_cutoff:
                    if (fname, fval, label) not in mapping:
                        mapping[fname, fval, label] = len(mapping)

        if labels is None:
            labels = seen_labels
        return cls(labels, mapping, **options)


class GISEncoding(BinaryMaxentFeatureEncoding):

    def __init__(self, labels, mapping, unseen_features=False, alwayson_features=False, C=None):
        BinaryMaxentFeatureEncoding.__init__(self, labels, mapping, unseen_features, alwayson_features)
        if C is None:
            C = len(set(fname for (fname, fval, label) in mapping)) + 1
        self._C = C

    @property
    def C(self):
        return self._C

    def encode(self, featureset, label):
        encoding = BinaryMaxentFeatureEncoding.encode(self, featureset, label)
        base_length = BinaryMaxentFeatureEncoding.length(self)

        total = sum(v for (f, v) in encoding)
        if total >= self._C:
            raise ValueError('Correction feature is not high enough!')
        encoding.append((base_length, self._C - total))

        return encoding

    def length(self):
        return BinaryMaxentFeatureEncoding.length(self) + 1

    def describe(self, f_id):
        if f_id == BinaryMaxentFeatureEncoding.length(self):
            return 'Correction feature (%s)' % self._C
        else:
            return BinaryMaxentFeatureEncoding.describe(self, f_id)


def train_maxent_classifier_with_gis(train_toks, validation_toks=None, trace=3, encoding=None, labels=None, **cutoffs):
    cutoffs.setdefault('max_iter', 100)
    cutoffchecker = CutoffChecker(cutoffs)

    if encoding is None:
        encoding = GISEncoding.train(train_toks, labels=labels)

    if not hasattr(encoding, 'C'):
        raise TypeError('The GIS algorithm requires an encoding that defines C (e.g., GISEncoding).')

    Cinv = 1.0 / encoding.C
    empirical_fcount = calculate_empirical_fcount(train_toks, encoding)
    unattested = set(np.nonzero(empirical_fcount == 0)[0])

    weights = np.zeros(len(empirical_fcount), 'd')
    for fid in unattested:
        weights[fid] = np.NINF
    classifier = ConditionalExponentialClassifier(encoding, weights)

    log_empirical_fcount = np.log2(empirical_fcount)
    del empirical_fcount

    if trace > 0:
        print('  ==> Training (%d iterations)' % cutoffs['max_iter'])
    if trace > 2:
        print()
        if validation_toks:
            print('      Iteration    Log Likelihood    Accuracy            Val LL      Val Acc')
            print('      ----------------------------------------------------------------------')
        else:
            print('      Iteration    Log Likelihood    Accuracy')
            print('      ---------------------------------------')

    try:
        while True:
            if trace > 2:
                ll = cutoffchecker.ll or log_likelihood(classifier, train_toks)
                acc = cutoffchecker.acc or accuracy(classifier, train_toks)
                iternum = cutoffchecker.iter
                if validation_toks:
                    val_ll = log_likelihood(classifier, validation_toks)
                    val_acc = accuracy(classifier, validation_toks)
                    print('     %9d    %14.5f    %9.3f    %14.5f    %9.3f' % (iternum, ll, acc, val_ll, val_acc))
                else:
                    print('     %9d    %14.5f    %9.3f' % (iternum, ll, acc))

            estimated_fcount = calculate_estimated_fcount(classifier, train_toks, encoding)

            for fid in unattested:
                estimated_fcount[fid] += 1
            log_estimated_fcount = np.log2(estimated_fcount)
            del estimated_fcount

            weights = classifier.weights()
            weights += (log_empirical_fcount - log_estimated_fcount) * Cinv
            classifier.set_weights(weights)

            if cutoffchecker.check(classifier, train_toks):
                break

    except KeyboardInterrupt:
        print('      Training stopped: keyboard interrupt')
    except:
        raise

    if trace > 2:
        ll = log_likelihood(classifier, train_toks)
        acc = accuracy(classifier, train_toks)
        if validation_toks:
            val_ll = log_likelihood(classifier, validation_toks)
            val_acc = accuracy(classifier, validation_toks)
            print('         Final    %14.5f    %9.3f    %14.5f    %9.3f' % (ll, acc, val_ll, val_acc))
        else:
            print('         Final    %14.5f    %9.3f' % (ll, acc))

    return classifier


def calculate_empirical_fcount(train_toks, encoding):
    fcount = np.zeros(encoding.length(), 'd')

    for tok, label in train_toks:
        for (index, val) in encoding.encode(tok, label):
            fcount[index] += val

    return fcount


def calculate_estimated_fcount(classifier, train_toks, encoding):
    fcount = np.zeros(encoding.length(), 'd')

    for tok, label in train_toks:
        pdist = classifier.prob_classify(tok)
        for label in pdist.samples():
            prob = pdist.prob(label)
            for (fid, fval) in encoding.encode(tok, label):
                fcount[fid] += prob * fval

    return fcount


def train_maxent_classifier_with_iis(train_toks, validation_toks=None, trace=3, encoding=None, labels=None, **cutoffs):
    cutoffs.setdefault('max_iter', 100)
    cutoffchecker = CutoffChecker(cutoffs)

    if encoding is None:
        encoding = BinaryMaxentFeatureEncoding.train(train_toks, labels=labels)

    empirical_ffreq = calculate_empirical_fcount(train_toks, encoding) / len(train_toks)

    nfmap = calculate_nfmap(train_toks, encoding)
    nfarray = np.array(sorted(nfmap, key=nfmap.__getitem__), 'd')
    nftranspose = np.reshape(nfarray, (len(nfarray), 1))

    unattested = set(np.nonzero(empirical_ffreq == 0)[0])

    weights = np.zeros(len(empirical_ffreq), 'd')
    for fid in unattested:
        weights[fid] = np.NINF
    classifier = ConditionalExponentialClassifier(encoding, weights)

    if trace > 0:
        print('  ==> Training (%d iterations)' % cutoffs['max_iter'])
    if trace > 2:
        print()
        if validation_toks:
            print('      Iteration    Log Likelihood    Accuracy            Val LL      Val Acc')
            print('      ----------------------------------------------------------------------')
        else:
            print('      Iteration    Log Likelihood    Accuracy')
            print('      ---------------------------------------')

    try:
        while True:
            if trace > 2:
                ll = cutoffchecker.ll or log_likelihood(classifier, train_toks)
                acc = cutoffchecker.acc or accuracy(classifier, train_toks)
                iternum = cutoffchecker.iter
                if validation_toks:
                    val_ll = log_likelihood(classifier, validation_toks)
                    val_acc = accuracy(classifier, validation_toks)
                    print('     %9d    %14.5f    %9.3f    %14.5f    %9.3f' % (iternum, ll, acc, val_ll, val_acc))
                else:
                    print('     %9d    %14.5f    %9.3f' % (iternum, ll, acc))

            deltas = calculate_deltas(
                train_toks,
                classifier,
                unattested,
                empirical_ffreq,
                nfmap,
                nfarray,
                nftranspose,
                encoding,
            )

            weights = classifier.weights()
            weights += deltas
            classifier.set_weights(weights)

            if cutoffchecker.check(classifier, train_toks):
                break

    except KeyboardInterrupt:
        print('      Training stopped: keyboard interrupt')
    except:
        raise

    if trace > 2:
        ll = log_likelihood(classifier, train_toks)
        acc = accuracy(classifier, train_toks)
        if validation_toks:
            val_ll = log_likelihood(classifier, validation_toks)
            val_acc = accuracy(classifier, validation_toks)
            print('         Final    %14.5f    %9.3f    %14.5f    %9.3f' % (ll, acc, val_ll, val_acc))
        else:
            print('         Final    %14.5f    %9.3f' % (ll, acc))

    return classifier


def calculate_nfmap(train_toks, encoding):
    nfset = set()
    for tok, _ in train_toks:
        for label in encoding.labels():
            nfset.add(sum(val for (id, val) in encoding.encode(tok, label)))
    return dict((nf, i) for (i, nf) in enumerate(nfset))


def calculate_deltas(
        train_toks,
        classifier,
        unattested,
        ffreq_empirical,
        nfmap,
        nfarray,
        nftranspose,
        encoding,
):
    NEWTON_CONVERGE = 1e-12
    MAX_NEWTON = 300

    deltas = np.ones(encoding.length(), 'd')
    A = np.zeros((len(nfmap), encoding.length()), 'd')

    for tok, label in train_toks:
        dist = classifier.prob_classify(tok)

        for label in encoding.labels():
            feature_vector = encoding.encode(tok, label)
            nf = sum(val for (id, val) in feature_vector)
            for (id, val) in feature_vector:
                A[nfmap[nf], id] += dist.prob(label) * val
    A /= len(train_toks)

    for rangenum in range(MAX_NEWTON):
        nf_delta = np.outer(nfarray, deltas)
        exp_nf_delta = 2 ** nf_delta
        nf_exp_nf_delta = nftranspose * exp_nf_delta
        sum1 = np.sum(exp_nf_delta * A, axis=0)
        sum2 = np.sum(nf_exp_nf_delta * A, axis=0)

        for fid in unattested:
            sum2[fid] += 1

        deltas -= (ffreq_empirical - sum1) / -sum2
        n_error = np.sum(abs((ffreq_empirical - sum1))) / np.sum(abs(deltas))
        if n_error < NEWTON_CONVERGE:
            return deltas

    return deltas
