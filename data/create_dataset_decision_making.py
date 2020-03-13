import argparse
import heapq
import logging
import multiprocessing
import os
import pickle
import random
from collections import OrderedDict
from ctypes import c_ulong, c_float
from itertools import combinations
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from queue import Empty
from time import time

import numpy as np
import sys

import sklearn
from sklearn.model_selection import train_test_split
from torch.nn import CosineSimilarity

from data.bug_report_database import BugReportDatabase
from data.create_dataset_deshmukh import savePairs, saveTriplets
from data.bug_dataset import BugDataset
from data.preprocessing import concatenateSummaryAndDescription
from util.data_util import createChunks


class RandomNonDuplicateGenerator(object):

    def __init__(self, bugs):
        self.bugs = bugs
        self.generatedNeg = set()

    @staticmethod
    def randBugId(list):
        return list[random.randint(0, len(list) - 1)]

    def generateNegativePair(self, n, masterIdById):
        p = 0

        while p < n:
            bugId = self.randBugId(bugs)['bug_id']
            bugId2 = self.randBugId(bugs)['bug_id']

            if bugId == bugId2:
                continue

            # Check if bug is not in the same master set
            if masterIdById[bugId] == masterIdById[bugId2]:
                continue

            # Check if that a negative example was already generated for that bug
            if bugId > bugId2:
                pair = (bugId, bugId2, -1)
            else:
                pair = (bugId2, bugId, -1)

            if pair in self.generatedNeg:
                continue

            yield pair
            self.generatedNeg.add(pair)
            p += 1

    def generateNegativeExample(self, n, bug, masterSet):
        p = 0
        bugId = bug['bug_id']

        while p < n:
            bugId2 = self.randBugId(self.bugIds)['bug_id']

            if bugId == bugId2:
                continue

            # Check if bug is not in the same master set
            if bugId2 in masterSet:
                continue

            # Check if that a negative example was already generated for that bug
            if bugId > bugId2:
                pair = (bugId, bugId2)
            else:
                pair = (bugId, bugId2)

            if pair in self.generatedNeg:
                continue

            yield bugId2
            self.generatedNeg.add(pair)
            p += 1


def generate_triplets(pairs, n, masterIdById):
    positive_pairs = []
    bug_ids = set()

    for query, cand, label in pairs:
        if label == 1:
            positive_pairs.append((query, cand))

        bug_ids.add(query)
        bug_ids.add(cand)

    bug_ids = list(bug_ids)

    triplets = []

    for anchor, dup in positive_pairs:
        for _ in range(n):
            master_id1 = masterIdById[anchor]
            master_id2 = master_id1

            while master_id1 == master_id2:
                negative = bug_ids[random.randint(0, len(bug_ids) - 1)]
                master_id2 = masterIdById[negative]

            triplets.append((anchor, dup, negative))

    for b, p, n in triplets:
        if masterIdByBugId[b] != masterIdByBugId[p]:
            print('Triplets: Positive bug is not correct! (%s,%s)' % (b, p))
            sys.exit(-1)

        if masterIdByBugId[b] == masterIdByBugId[n]:
            print('Triplets: Negative bug is not correct! (%s,%s)' % (b, n))
            sys.exit(-1)

    return triplets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--database', required=True, help="Json file that contains all reports.")
    parser.add_argument('--n', dest='n', required=True, type=int, help="rate of positive and negative pares")
    parser.add_argument('--round', dest='round', required=True, type=int, help="rate of positive and negative pares")
    parser.add_argument('--validation_size', type=float, default=0.1, help="where will save the training data")
    parser.add_argument('--test_size', type=float, default=0.2, help="where will save the training data")

    logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)

    bugReportDatabase = BugReportDatabase.fromJson(args.database)

    masterSetById = bugReportDatabase.getMasterSetById()
    masterIdById = bugReportDatabase.getMasterIdByBugId()
    bugs = bugReportDatabase.bugList

    generator = RandomNonDuplicateGenerator(bugs)

    total = len(masterSetById)
    current = 0
    last = time()
    emptySet = set()

    pairs = []

    logger.info("Generating triplets and pairs")
    for masterId, masterset in masterSetById.items():
        for duplicatePair in combinations(masterset, 2):
            pairs.append((duplicatePair[0], duplicatePair[1], 1))
            pairs.extend(generator.generateNegativePair(args.n, masterIdById))

        if current != 0 and current % 400 == 0:
            c = time()
            logger.info("Processed %d mastersets of %d. Time: %f" % (current, total, c - last))
            last = time()

        current += 1

    #

    # Check if there are duplicate pairs and triplets
    nPairs = len(pairs)
    pairs_set = set()
    labels = []

    for anchor, dup, label in pairs:
        labels.append(label)
        pairs_set.add((min(anchor, dup), max(anchor, dup)))

    labels = np.asarray(labels)
    n_pos = (labels == 1).sum()
    n_neg = (labels == -1).sum()

    logger.info('Number of positives={} , Number of Negatives={} Ratio={}'.format(n_pos,n_neg,n_neg/n_pos))

    if len(pairs_set) != nPairs:
        print('A duplicate pair was found!')
        sys.exit(-1)

    # Check if the bugs were labeled wrongly
    masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

    for b1, b2, l in pairs:
        if l == 1 and masterIdByBugId[b1] != masterIdByBugId[b2]:
            print('Positive bug is not correct! (%s,%s)' % (b1, b2))
            sys.exit(-1)

        if l == -1 and masterIdByBugId[b1] == masterIdByBugId[b2]:
            print('Negative bug is not correct! (%s,%s)' % (b1, b2))
            sys.exit(-1)

    total = len(pairs)

    folder = os.path.split(args.database)[0]

    trainPairs, testPairs = train_test_split(pairs, test_size=args.test_size)
    trainSplitPairs, validationPairs = train_test_split(trainPairs, test_size=args.validation_size)

    logger.info('--------------------------------------------')
    logger.info("Generating training triplets and pairs")

    savePairs(trainSplitPairs, "%s/training_split_classification_pairs_%d_round_%d.txt" % (folder, args.n, args.round))
    saveTriplets(generate_triplets(trainSplitPairs, args.n, masterIdById),
                 "%s/training_split_classification_triplets_%d_round_%d.txt" % (folder, args.n, args.round))

    logger.info('--------------------------------------------')
    logger.info("Generating validation triplets and pairs")
    savePairs(validationPairs, "%s/validation_classification_pairs_%d_round_%d.txt" % (folder, args.n, args.round))

    logger.info('--------------------------------------------')
    logger.info("Generating test triplets and pairs")
    savePairs(testPairs, "%s/test_classification_pairs_%d_round_%d.txt" % (folder, args.n, args.round))

    logger.info(
        "Training duplicate pairs: %d (%.2f); Validation duplicate pairs: %d (%.2f); Test duplicate pairs: %d (%.2f)" %
        (len(trainSplitPairs), len(trainSplitPairs) / total * 100,
         len(validationPairs), len(validationPairs) / total * 100,
         len(testPairs), len(testPairs) / total * 100))

    logger.info("Total Pairs: %d" % len(pairs))
