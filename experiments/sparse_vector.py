import argparse
import logging
import pickle
import random
import string

import numpy as np

from nltk import TreebankWordTokenizer, SnowballStemmer, WhitespaceTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

from data.bug_report_database import BugReportDatabase
from data.bug_dataset import BugDataset
from data.preprocessing import concatenateSummaryAndDescription, ClassicalPreprocessing, MultiLineTokenizer, \
    StripPunctuactionFilter, DectectNotUsualWordFilter, TransformNumberToZeroFilter


def compareSimilarity(file, label, margin=0.1):
    total = 0
    nBiggerMargin = 0
    nBigger = 0
    f = open(file, 'r')

    for l in f:
        bug, pos, neg = l.strip().split(',')

        bugText = concatenateSummaryAndDescription(bugReportDataset.getBug(bug))
        posText = concatenateSummaryAndDescription(bugReportDataset.getBug(pos))
        negText = concatenateSummaryAndDescription(bugReportDataset.getBug(neg))

        ftrsBug, ftrsPos, ftrsNeg = vectorizer.transform([bugText, posText, negText])

        simPos = cosine_similarity(ftrsBug, ftrsPos)
        simNeg = cosine_similarity(ftrsBug, ftrsNeg)

        if simPos - simNeg >= margin:
            nBiggerMargin += 1

        if simPos > simNeg:
            nBigger += 1

        total += 1

    print("%s - triplets which the difference between positive and negative is bigger than %d: %.3f (%d/%d)" % (
        label, margin, nBiggerMargin * 100.0 / total, nBiggerMargin, total))
    print("%s - triplets which the positive similarity is bigger than the negative: %.3f (%d/%d)" % (
        label, nBigger * 100.0 / total, nBigger, total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', )
    parser.add_argument('--validation', )
    parser.add_argument('--bug_dataset')
    parser.add_argument('--type', default="tfidf", help="")
    parser.add_argument('--stemmer', action='store_true')
    parser.add_argument('--bigram', action='store_true')
    parser.add_argument('--trigram', action='store_true')
    parser.add_argument('--max_features', type=int)
    parser.add_argument('--load')
    parser.add_argument('--save')
    parser.add_argument('--threshold', type=float, help="")
    parser.add_argument('--is_pairs', action='store_true', help="")
    parser.add_argument('--calculate_cosine', action='store_true', help="")
    parser.add_argument('--training_triplets')
    parser.add_argument('--validation_triplets')
    parser.add_argument('--space_tokenize', action='store_true')

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    args = parser.parse_args()

    print(args)

    bugReportDataset = BugReportDatabase.fromJson(args.bug_dataset)

    trainingBugs = set()
    triplets = []
    if args.training:

        if args.is_pairs:
            logger.info("Reading training with pairs")
            f = open(args.training, 'r')

            for l in f:
                bugId1, bugId2, label = l.strip().split(',')

                trainingBugs.add(bugId1)
                trainingBugs.add(bugId2)
        else:
            logger.info("Reading training")
            bugDataset = BugDataset(args.training)
            trainingBugs.update(bugDataset.bugIds)


    logger.info("Preprocessing and fitting data")
    trainingText = []

    for bugId in trainingBugs:
        bugReport = bugReportDataset.getBug(bugId)
        text = concatenateSummaryAndDescription(bugReport)
        trainingText.append(text)

    if args.load:
        logger.info('Loading  object')
        vectorizer = pickle.load(open(args.load, 'rb'))
    else:
        tokenizer = WhitespaceTokenizer() if args.space_tokenize else MultiLineTokenizer()
        stemmer = SnowballStemmer('english', ignore_stopwords=True) if args.stemmer else None
        stopWords = set(stopwords.words('english') + list(string.punctuation) + ["n't", "'t", ])
        filters = [TransformNumberToZeroFilter(), StripPunctuactionFilter(), DectectNotUsualWordFilter()]
        tokenizerStemmer = ClassicalPreprocessing(tokenizer, stemmer, stopWords, filters)

        logger.info("Using %s to tokenize" % tokenizer.__class__.__name__)

        if args.trigram:
            ngramRange = (1, 3)
        elif args.bigram:
            ngramRange = (1, 2)
        else:
            ngramRange = (1, 1)

        if args.type == 'tfidf':
            logger.info('TF-IDF')
            vectorizer = TfidfVectorizer(tokenizer=tokenizerStemmer.preprocess, stop_words=None, lowercase=True,
                                         analyzer='word', max_features=args.max_features, ngram_range=ngramRange)
        elif args.type == 'binary':
            logger.info('BINARY')
            vectorizer = CountVectorizer(tokenizer=tokenizerStemmer.preprocess, stop_words=None, lowercase=True,
                                         analyzer='word', max_features=args.max_features, ngram_range=ngramRange,
                                         binary=True)
        else:
            vectorizer = None

        vectorizer.fit(trainingText)
        f = open('/home/irving/lexicon_%d.txt' % random.randint(0, 99999999), 'w')
        l = [''] * len(vectorizer.vocabulary_)

        for k, v in vectorizer.vocabulary_.items():
            l[v] = k

        f.write(str(args))
        f.write('\n')

        for wo in l:
            f.write(wo)
            f.write('\n')

    if args.save:
        logger.info('Saving tf-idf object')
        pickle.dump(vectorizer, open(args.save, 'wb'))

    logger.info('Vocabulary size: %d' % len(vectorizer.vocabulary_))

    nBiggerMargin = 0
    nBigger = 0

    if args.calculate_cosine:
        logger.info("Compare cosine difference")

        if args.training_triplets:
            compareSimilarity(args.training_triplets, 'training')

        if args.validation_triplets:
            compareSimilarity(args.validation_triplets, 'validation')

    if args.validation:
        logger.info("Evaluating Method")
        f = open(args.validation, 'r')
        targets = []
        predictions = []
        validationText = []

        for l in f:
            bugId1, bugId2, label = l.strip().split(',')

            bug1 = concatenateSummaryAndDescription(bugReportDataset.getBug(bugId1))
            bug2 = concatenateSummaryAndDescription(bugReportDataset.getBug(bugId2))

            ftrsBug1, ftrsBug2 = vectorizer.transform([bug1, bug2])

            similarity = cosine_similarity(ftrsBug1, ftrsBug2)

            if similarity > args.threshold:
                predictions.append(1)
            else:
                predictions.append(0)

            targets.append(max(0, int(label)))

        accum = accuracy_score(targets, predictions, normalize=False)
        acc = accum / len(targets)
        prec, recall, f1, _ = precision_recall_fscore_support(targets, predictions)

        print("Accuracy: %.3f (%d/%d)" % (acc * 100, accum, len(targets)))
        print("Precision: {}\tRecall: {}\tF1:{}".format(list(np.around(prec * 100, decimals=3)),
                                                        list(np.around(recall * 100, decimals=3)),
                                                        list(np.around(f1 * 100, decimals=3))))
