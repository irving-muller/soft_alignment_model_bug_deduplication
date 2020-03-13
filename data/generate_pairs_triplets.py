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
from torch.nn import CosineSimilarity

from data.bug_report_database import BugReportDatabase
from data.create_dataset_deshmukh import savePairs, saveTriplets
from data.bug_dataset import BugDataset
from data.preprocessing import concatenateSummaryAndDescription
from util.data_util import createChunks


class RandomNonDuplicateGenerator(object):

    def __init__(self, bugIds):
        self.bugIds = list(bugIds)
        self.generatedNeg = set()

    @staticmethod
    def randBugId(list):
        return list[random.randint(0, len(list) - 1)]

    def generateNegativeExample(self, n, bugId, masterSet):
        p = 0
        while p < n:
            bug2 = self.randBugId(self.bugIds)

            if bugId == bug2:
                continue

            # Check if bug is not in the same master set
            if bug2 in masterSet:
                continue

            # Check if that a negative example was already generated for that bug
            if bugId > bug2:
                pair = (bugId, bug2)
            else:
                pair = (bugId, bug2)

            if pair in self.generatedNeg:
                continue

            yield bug2
            self.generatedNeg.add(pair)
            p += 1


class SimScore(object):

    def __init__(self, bugId, score):
        self.bugId = bugId
        self.score = score

    def __eq__(self, other):
        return self.score == other.score

    def __ne__(self, other):
        return self.score != other.score

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __repr__(self):
        return "%s %s" % (self.first, self.last)


class VectorizerGenerator(object):

    def __init__(self, bugReportDatabase, bugIds, vectorizerClass, bugToCache, mastersetByBugId, normalize):
        self.bugReportDatabase = bugReportDatabase
        self.bugIds = bugIds
        self.vectorizerClass = vectorizerClass

        bugTexts = [concatenateSummaryAndDescription(bugReportDatabase.getBug(bugId)) for bugId in bugIds]
        self.logger = logging.getLogger(__name__)

        self.logger.info("Transforming text to vector")
        vectors = self.vectorizerClass.transform(bugTexts)

        if normalize:
            self.logger.info("Normalizing vectors to length 1")
            self.matrixRep = sklearn.preprocessing.data.normalize(vectors)
        else:
            self.matrixRep = vectors

        self.sparseRepByBugId = {}
        for bugId, representation in zip(self.bugIds, self.matrixRep):
            self.sparseRepByBugId[bugId] = representation.T

        self.similarityIterByBugId = {}

        # Cache the similarity list of the bugs
        self.logger.info("Starting to cache the similarity list")

        def parallel(chunk, queue, index):
            logger = logging.getLogger()
            logger.info(
                "Process %s started to compute the similarity for %d duplicate bugs. Start idx: %d" % (
                    os.getpid(), len(chunk), index))

            output = []
            start = time()
            for idx, bugId in enumerate(chunk):
                bugId = str(bugId)
                simList = self.generateSimilarityList(bugId, mastersetByBugId[bugId], iterator=False)
                output.append((bugId, simList))

                if (idx + 1) % 100 == 0:
                    t = time() - start
                    self.logger.info("%s computed similarity list for %d of %d in %f seconds" % (
                    os.getpid(), idx + 1, len(chunk), t))

            queue.put(output)

        q = Queue()
        nProcesses = 6
        processes = []
        for idx, chunk in enumerate(createChunks(bugToCache, nProcesses)):
            arr = RawArray(c_ulong, [int(bugId) for bugId in chunk])
            processes.append(multiprocessing.Process(target=parallel, args=(arr, q, idx)))

        for p in processes:
            p.start()

        count = 0

        while True:
            try:
                for bugId, simList in q.get():
                    self.similarityIterByBugId[bugId] = iter(simList)

                count += 1

                if count == len(processes):
                    break
            except Empty as e:
                pass

    def iterateSimList(self, cosineSim, masterSet):
        for otherBugId, cosineSim in zip(self.bugIds, cosineSim):
            if otherBugId not in masterSet:
                yield SimScore(otherBugId, cosineSim[0])

    def generateSimilarityList(self, bugId, masterSet, iterator=True):
        bugRep = self.sparseRepByBugId[bugId]
        similarity = self.matrixRep.dot(bugRep).toarray()
        nMostSimilar = heapq.nlargest(300, self.iterateSimList(similarity, masterSet))

        if iterator:
            return iter(nMostSimilar)

        return nMostSimilar

    def generateNegativeExample(self, n, bugId, masterSet):
        # Search if the bug Id was used before
        similarityIter = self.similarityIterByBugId.get(bugId, None)

        if similarityIter is None:
            # self.logger.debug("Missed bug(%s)! Calculating similarity list" % bugId)
            similarityIter = self.generateSimilarityList(bugId, masterSet)
            self.similarityIterByBugId[bugId] = similarityIter

        for _ in range(n):
            yield similarityIter.__next__().bugId


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--bug_data', required=True, help="")
    parser.add_argument('--dataset', required=True, help="")
    parser.add_argument('--n', required=True, type=int, help="")
    parser.add_argument('--type', required=True, help="")
    parser.add_argument('--aux_file', help="")
    parser.add_argument('--model', help="")

    logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)

    bugDataset = BugDataset(args.dataset)
    bugReportDatabase = BugReportDatabase.fromJson(args.bug_data)

    bugIds = bugDataset.bugIds
    duplicateBugs = bugDataset.duplicateIds

    if args.aux_file:
        '''
        In our methodology, we compare new bug with all the previously bugs that are in the database.
        To better generate pairs and triplets, we use the bugs that were reported before the ones 
        from the validation.
        '''
        auxBugDataset = BugDataset(args.aux_file)
        bugsFromMainFile = list(bugIds)
        bugsFromMainFileSet = set(bugIds)

        for auxBugIds in auxBugDataset.bugIds:
            bugIds.append(auxBugIds)

        useAuxFile = True
    else:
        bugsFromMainFile = list(bugIds)
        bugsFromMainFileSet = set(bugIds)
        useAuxFile = False

    # Insert all master ids to the bug id list. The master can be in another file and we need them at least to create a 1 pair
    masterSetById = bugReportDatabase.getMasterSetById(bugIds)

    for masterId in masterSetById.keys():
        bugIds.append(masterId)

    masterIdByBugId = bugReportDatabase.getMasterIdByBugId(bugIds)

    # Convert to set to avoid duplicate bug ids
    bugIds = set(bugIds)

    triplets = []
    pairs = []

    if args.type == 'random':
        generator = RandomNonDuplicateGenerator(bugIds)
    elif args.type in set(['tfidf', 'binary']):
        # Insert imports to load TfIdfVectorizer class
        from nltk import TreebankWordTokenizer, SnowballStemmer
        from nltk.corpus import stopwords
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from data.BugDataset import BugDataset
        from data.preprocessing import ClassicalPreprocessing, MultiLineTokenizer, \
            StripPunctuactionFilter, DectectNotUsualWordFilter, TransformNumberToZeroFilter

        tfIdfVectorizer = pickle.load(open(args.model, 'rb'))
        mastersetByBugId = {}
        bugToCache = []
        for masterset in masterSetById.values():
            for duplicatePair in combinations(masterset, 2):
                if not useAuxFile or duplicatePair[0] not in bugsFromMainFileSet:
                    bugToCache.append(duplicatePair[0])
                    mastersetByBugId[duplicatePair[0]] = masterset

        normalize = True if args.type == 'tfidf' else False
        generator = VectorizerGenerator(bugReportDatabase, bugIds, tfIdfVectorizer, bugToCache, mastersetByBugId,
                                        normalize)
    else:
        raise Exception('%s is not available option' % args.type)

    total = len(masterSetById)
    current = 0
    last = time()
    emptySet = set()
    listBugIds = list(bugIds)

    logger.info("Generating triplets and pairs")
    for masterId, masterset in masterSetById.items():
        for duplicatePair in combinations(masterset, 2):
            if useAuxFile:
                if duplicatePair[0] not in bugsFromMainFileSet and duplicatePair[1] not in bugsFromMainFileSet:
                    # Duplicate pairs that none of the bugs are in the file input were already generated in training dataset
                    continue

            for negativeExample in generator.generateNegativeExample(args.n, duplicatePair[0], masterset):
                triplets.append((duplicatePair[0], duplicatePair[1], negativeExample))

            #todo: I disabled pair generation for now. We have to find a best way to generate good negative pair when we are using tf-idf
            # pairs.append((duplicatePair[0], duplicatePair[1], 1))
            # Generating negative pairs. First bugs of the pairs are randomly generated.
            # for _ in range(args.n):
            #     firstBug = listBugIds[random.randint(0, len(listBugIds) - 1)]
            #     firstMasterSet = masterSetById.get(masterIdByBugId[firstBug])
            #
            #     if firstMasterSet is None:
            #         firstMasterSet = set()
            #
                # todo: Improve Code
                # for secondBug in generator.generateNegativeExample(1, firstBug, firstMasterSet):
                #     pairs.append((firstBug, secondBug, -1))


        if current != 0 and current % 400 == 0:
            c = time()
            logger.info("Processed %d mastersets of %d. Time: %f" % (current, total, c - last))
            last = time()

        current += 1

    pairs = set()

    for anchor, dup, nondup in triplets:
        pairs.add((anchor, dup, 1))
        pairs.add((anchor, nondup, -1))

    # Check if there are duplicate pairs and triplets
    nPairs = len(pairs)

    if len(set(pairs)) != nPairs:
        print('A duplicate pair was found!')
        sys.exit(-1)

    nTriplets = len(triplets)

    if len(set(triplets)) != nTriplets:
        print('A duplicate triplet was found!')
        sys.exit(-1)


    # Check if the bugs were labeled wrongly
    masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

    for b,p,n in triplets:
        if masterIdByBugId[b] != masterIdByBugId[p]:
            print('Triplets: Positive bug is not correct! (%s,%s)' % (b,p))
            sys.exit(-1)

        if masterIdByBugId[b] == masterIdByBugId[n]:
            print('Triplets: Negative bug is not correct! (%s,%s)' % (b,n))
            sys.exit(-1)

    for b1,b2,l in pairs:
        if l == 1 and masterIdByBugId[b1] != masterIdByBugId[b2]:
            print('Positive bug is not correct! (%s,%s)' % (b1,b2))
            sys.exit(-1)

        if l == -1 and masterIdByBugId[b1] == masterIdByBugId[b2]:
            print('Negative bug is not correct! (%s,%s)' % (b1,b2))
            sys.exit(-1)



    part1, part2 = os.path.splitext(args.dataset)

    name = os.path.splitext(os.path.split(args.model)[1])[0] if args.model else args.type

    savePairs(pairs, "%s_pairs_%s_%d%s" % (part1, name, args.n, part2))
    saveTriplets(triplets, "%s_triplets_%s_%d%s" % (part1, name, args.n, part2))

    nDupPairs = np.asarray([l if l == 1 else 0 for b1, b2, l in pairs]).sum()
    logger.info(
        '%d duplicate pairs\t%d non-duplicate pairs\t%d pairs' % (nDupPairs, len(pairs) - nDupPairs, len(pairs)))
    logger.info("Total triplets: %d" % len(triplets))
