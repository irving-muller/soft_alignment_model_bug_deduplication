"""
Train and test a classical models to the duplicate bug report detection.
"""
import argparse
import logging
import os

import pickle
from time import time

from nltk import TreebankWordTokenizer, SnowballStemmer
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from data.preprocessing import concatenateSummaryAndDescription, ClassicalPreprocessing, SunFeatureExtractor


def calculateIdfs(bugReportDatabase, classicProcessing, bugIds, field):
    trainingText = []

    for bugId in bugIds:
        bugReport = bugReportDatabase.getBug(bugId)

        if isinstance(field, (str)):
            trainingText.append(bugReport[field])
        else:
            trainingText.append(field(bugReport))
    tfIdfVectorizer = TfidfVectorizer(tokenizer=classicProcessing.preprocess, stop_words=None, lowercase=True,
                                      analyzer='word', ngram_range=(1, 2))
    tfIdfVectorizer.fit(trainingText)

    return tfIdfVectorizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bug_database', required=True)
    parser.add_argument('--training_reports')
    parser.add_argument('--idf_basename')
    parser.add_argument('--training_pairs', required=True)
    parser.add_argument('--save', required=True)
    parser.add_argument('--alg', required=True)

    parser.add_argument('--C', type=float, default=1.0)

    logging.basicConfig(format='%(asctime)s %(levelname)-4s %(message)s', level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S', )
    logger = logging.getLogger()
    args = parser.parse_args()

    print(args)

    bugReportDatabase = BugReportDatabase.fromJson(args.bug_database)

    descIdfFileName = args.idf_basename + '_description_tfidf.pk'
    sumIdfFileName = args.idf_basename + '_summary_tfidf.pk'
    bothIdfFileName = args.idf_basename + '_both_tfidf.pk'

    tokenizer = TreebankWordTokenizer()
    stemmer = SnowballStemmer('english', ignore_stopwords=True)
    stopWords = set(stopwords.words('english'))

    classicalPreProcessing = ClassicalPreprocessing(tokenizer, stemmer, stopWords)

    if args.training_reports is not None:
        bugSetDataset = BugDataset(args.training_reports)
        bugIds = []

        for idx in range(bugSetDataset.end):
            bugIds.append(bugReportDatabase.getBugByIndex(idx)['bug_id'])

        if os.path.isfile(descIdfFileName):
            logger.warning("Idf file %s exists and it will be overwritten." % descIdfFileName)

        logger.info("Computing and saving idf of the description in the training")
        descTfidf = calculateIdfs(bugReportDatabase, classicalPreProcessing, bugIds, 'description')
        pickle.dump(descTfidf, open(descIdfFileName, 'wb'))

        if os.path.isfile(sumIdfFileName):
            logger.warning("Idf file %s exists and it will be overwritten." % sumIdfFileName)

        logger.info("Computing idf of the summary in the training")
        sumTfidf = calculateIdfs(bugReportDatabase, classicalPreProcessing, bugIds, 'short_desc')
        pickle.dump(sumTfidf, open(sumIdfFileName, 'wb'))

        if os.path.isfile(bothIdfFileName):
            logger.warning("Idf file %s exists and it will be overwritten." % bothIdfFileName)

        logger.info("Computing idf of the summary+description in the training")
        bothTfidf = calculateIdfs(bugReportDatabase, classicalPreProcessing, bugIds, concatenateSummaryAndDescription)
        pickle.dump(bothTfidf, open(bothIdfFileName, 'wb'))
    else:
        logger.info('Loading tf-idf of description')
        descTfidf = pickle.load(open(descIdfFileName, 'rb'))

        logger.info('Loading tf-idf of summary')
        sumTfidf = pickle.load(open(sumIdfFileName, 'rb'))

        logger.info('Loading tf-idf of summary+description')
        bothTfidf = pickle.load(open(bothIdfFileName, 'rb'))

    if args.alg == 'svm':
        model = SVC(probability=True, kernel='linear', C=args.C)

    featureExtractor = SunFeatureExtractor(classicalPreProcessing, descTfidf, sumTfidf, bothTfidf)

    f = open(args.training_pairs, 'r')

    X = []
    y = []

    logger.info("Extracting features")
    for l in f:
        bugId1, bugId2, label = l.strip().split(',')
        X.append(featureExtractor.extract(bugReportDatabase.getBug(bugId1), bugReportDatabase.getBug(bugId2)))
        y.append(max(0, int(label)))

    logger.info("Training")
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model.fit(X_scaled, y)
    logger.info("Saving Model")

    pickle.dump({'model': model, 'extractor': featureExtractor, 'scaler': scaler}, open(args.save, 'wb'))
