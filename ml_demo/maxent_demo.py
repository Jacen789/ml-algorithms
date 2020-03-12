# -*- coding: utf-8 -*-

import os
import random

from ml.maxent.classifier import MaxentClassifier
from ml.maxent.util import accuracy

from nltk.corpus import names

here = os.path.dirname(os.path.abspath(__file__))


def names_demo_features(name):
    features = {}
    features['alwayson'] = True
    features['startswith'] = name[0].lower()
    features['endswith'] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['count(%s)' % letter] = name.lower().count(letter)
        features['has(%s)' % letter] = letter in name.lower()
    return features


def demo1():
    # Construct a list of classified names, using the names corpus.
    namelist = [(name, 'male') for name in names.words('male.txt')] + [
        (name, 'female') for name in names.words('female.txt')
    ]

    # Randomly split the names into a test & train set.
    random.seed(123456)
    random.shuffle(namelist)
    train = namelist[:5000]
    test = namelist[5000:5500]

    # Train up a classifier.
    print('Training classifier...')
    classifier = MaxentClassifier.train(
        [(names_demo_features(n), g) for (n, g) in train],
        algorithm='iis',
        max_iter=10
    )

    # Run the classifier on the test data.
    print('Testing classifier...')
    acc = accuracy(classifier, [(names_demo_features(n), g)
                                for (n, g) in test])
    print('Accuracy: %6.4f' % acc)

    # For classifiers that can find probabilities, show the log
    # likelihood and some sample probability distributions.
    test_featuresets = [names_demo_features(n) for (n, g) in test]
    pdists = classifier.prob_classify_many(test_featuresets)
    ll = [pdist.logprob(gold) for ((name, gold), pdist) in zip(test, pdists)]
    print('Avg. log likelihood: %6.4f' % (sum(ll) / len(test)))
    print()
    print('Unseen Names      P(Male)  P(Female)\n' + '-' * 40)
    for ((name, gender), pdist) in list(zip(test, pdists))[:5]:
        if gender == 'male':
            fmt = '  %-15s *%6.4f   %6.4f'
        else:
            fmt = '  %-15s  %6.4f  *%6.4f'
        print(fmt % (name, pdist.prob('male'), pdist.prob('female')))


def demo2():
    import pickle
    import pprint

    # Construct a list of classified names, using the names corpus.
    namelist = [(name, 'male') for name in names.words('male.txt')] + [
        (name, 'female') for name in names.words('female.txt')
    ]

    # Randomly split the names into a test & train set.
    random.seed(123456)
    random.shuffle(namelist)
    train = namelist[:5000]
    test = namelist[5000:5500]

    model_file = os.path.join(here, 'model.pkl')
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            classifier = pickle.load(f)
    else:
        # Train up a classifier.
        print('Training classifier...')
        classifier = MaxentClassifier.train(
            [(names_demo_features(n), g) for (n, g) in train],
            validation_toks=[(names_demo_features(n), g) for (n, g) in test],
            algorithm='gis',
            max_iter=10
        )
        with open(model_file, 'wb') as f:
            pickle.dump(classifier, f)

    # Run the classifier on the test data.
    print('Testing classifier...')
    acc = accuracy(classifier, [(names_demo_features(n), g)
                                for (n, g) in test])
    print('Accuracy: %6.4f' % acc)

    test_featuresets = [names_demo_features(n) for (n, g) in test]
    pprint.pprint(test_featuresets[0])
    classifier.explain(test_featuresets[0])


if __name__ == '__main__':
    demo2()
