import argparse
import logging
import os
import pickle
import random
import ujson
import sys

import math
from ctypes import c_ulong
from multiprocessing import Array, Queue
from multiprocessing.sharedctypes import RawArray
from queue import Empty
from time import time

import numpy as np
import resource
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

from data.bug_report_database import BugReportDatabase
from data.preprocessing import concatenateSummaryAndDescription
from experiments.sparse_vector import TokenizerStemmer

from nltk import TreebankWordTokenizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def loadData(filePath):
    f = open(filePath, 'r')
    bugIds = set()
    duplicateByBugId = {}
    pairs = []

    for l in f:
        l = l.strip()
        if len(l) == 0:
            break

        bug1Id, bug2Id, label = l.split(',')
        label = int(label)
        pairs.append((bug1Id, bug2Id, label))

        bugIds.add(bug1Id)
        bugIds.add(bug2Id)

        if label == 1:
            duplicateBug1List = duplicateByBugId.get(bug1Id, set())

            if len(duplicateBug1List) == 0:
                duplicateByBugId[bug1Id] = duplicateBug1List

            duplicateBug1List.add(bug2Id)

            duplicateBug2List = duplicateByBugId.get(bug2Id, set())

            if len(duplicateBug2List) == 0:
                duplicateByBugId[bug2Id] = duplicateBug2List

            duplicateBug2List.add(bug1Id)

    return bugIds, duplicateByBugId, pairs, ujson.loads(f.readline())['validations']


class Obj(object):

    def __init__(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


def predictDeepLearningModel(bugEmbeddingsById, validationPairs):
    batchSize = 1024
    predictions = []
    nBatches = math.ceil(float(len(validationPairs)) / batchSize)

    firstBugPairs = []
    secondBugPairs = []

    for bug1, bug2 in validationPairs:
        firstBugPairs.append(bugEmbeddingsById[bug1])
        secondBugPairs.append(bugEmbeddingsById[bug2])

    for batchIdx in range(nBatches):
        batchStart = batchIdx * batchSize
        bug1s = getVariable(torch.stack(firstBugPairs[batchStart: batchStart + batchSize]), args.cuda)
        bug2s = getVariable(torch.stack(secondBugPairs[batchStart: batchStart + batchSize]), args.cuda)

        if arguments.model == 'retrieval':
            predictionInput = [bug1s, bug2s]
        elif arguments.model == 'classification':
            predictionInput = model[1](bug1s, bug2s)

        output = predictionFunction(predictionInput).data.cpu().numpy()

        for pr in output:
            if isinstance(pr, (np.float32, np.uint8)):
                predictions.append(pr)
            else:
                predictions.append(pr[-1])

    return predictions


def parallel(start, duplicateBugs, q):
    logger = logging.getLogger()
    c = time()
    logger.info(
        "Process %s started to compute the similarity for %d duplicate bugs. Start idx: %d" % (os.getpid(), len(duplicateBugs), start))

    for i, db in enumerate(duplicateBugs):
        q.put([start + i, calculateSimiliratyScoreTFIDF(str(db), vectorByBug, bugIds)])

        if i % 20 == 0 and i != 0:
            logger.info("TF-IDF: Process %s processed %d Duplicate bug of %d in %f" % (
                os.getpid(), i, len(duplicateBugs), time() - c))
            c = time()

    q.put([-1, None])


def calculateSimiliratyScoreTFIDF(duplicateBug, vectorByBug, bugIds):
    batchSize = 1024
    nPairs = len(bugIds)
    nBatches = math.ceil(float(nPairs) / batchSize)

    bugEmbedding1 = vectorByBug[duplicateBug]
    similarityScores = []
    nbDim = bugEmbedding1.shape[1]

    for batchIdx in range(nBatches):
        batchStart = batchIdx * batchSize

        data1 = []
        indices1 = []
        ptrs1 = [0]

        data2 = []
        indices2 = []
        ptrs2 = [0]

        for otherBug in bugIds[batchStart: batchStart + batchSize]:
            data1.extend(bugEmbedding1.data)
            indices1.extend(bugEmbedding1.indices)
            ptrs1.append(len(indices1))

            bugEmbedding2 = vectorByBug[otherBug]

            data2.extend(bugEmbedding2.data)
            indices2.extend(bugEmbedding2.indices)
            ptrs2.append(len(indices2))

        matrix1 = csr_matrix((data1, indices1, ptrs1), shape=(len(ptrs1) - 1, nbDim))
        matrix2 = csr_matrix((data2, indices2, ptrs2), shape=(len(ptrs2) - 1, nbDim))

        score = cosine_similarity(matrix1, matrix2)

        for i in range(score.shape[0]):
            similarityScores.append(score[i][i])

    return similarityScores


def predictTFIDF(pairs):
    batchSize = 8192
    nPairs = len(pairs)
    nBatches = math.ceil(float(nPairs) / batchSize)

    similarityScores = []

    for batchIdx in range(nBatches):
        batchStart = batchIdx * batchSize

        data1 = []
        indices1 = []
        ptrs1 = [0]

        data2 = []
        indices2 = []
        ptrs2 = [0]

        for bug1, bug2 in pairs[batchStart: batchStart + batchSize]:
            bugEmbedding1 = vectorByBug[bug1]

            data1.extend(bugEmbedding1.data)
            indices1.extend(bugEmbedding1.indices)
            ptrs1.append(len(indices1))

            bugEmbedding2 = vectorByBug[bug2]

            data2.extend(bugEmbedding2.data)
            indices2.extend(bugEmbedding2.indices)
            ptrs2.append(len(indices2))

        nbDim = vectorByBug[bug1].shape[1]
        pairBug1 = csr_matrix((data1, indices1, ptrs1), shape=(len(ptrs1) - 1, nbDim))
        pairBug2 = csr_matrix((data2, indices2, ptrs2), shape=(len(ptrs2) - 1, nbDim))

        score = cosine_similarity(pairBug1, pairBug2)

        for i in range(score.shape[0]):
            similarityScores.append(score[i][i])

    return (np.asarray(similarityScores) > args.retrieval_threshold).astype(int)


def chunks(l, n):
    chunkSize = int(len(l) / n)
    remaining = len(l) % n
    chunks = []
    begin = 0

    for i in range(n):
        if remaining != 0:
            additional = 1
            remaining -= 1
        else:
            additional = 0

        end = begin + chunkSize + additional
        chunks.append(l[begin:end])
        begin = end

    return chunks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recall_ratio_k', nargs='+', required=True,
                        help="list of the values of k to be used in the recall ratio. If k is empty list so recall rate "
                             "is not calculated")
    parser.add_argument('--model', help="model")
    parser.add_argument('--model_type', help="model type")
    parser.add_argument('--bug_dataset', help="")
    parser.add_argument('--input', required=True)
    parser.add_argument('--retrieval_threshold', type=float, default=None, help="")
    parser.add_argument('--nb_processes', type=int, default=8, help="")
    parser.add_argument('--cuda', action="store_true", help="enable cuda.")

    logging.basicConfig(format='%(asctime)s %(levelname)-4s %(message)s', level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S', )

    logger = logging.getLogger()

    args = parser.parse_args()
    print(args)

    global bugIds
    args.recall_ratio_k = [int(k) for k in args.recall_ratio_k]
    bugIds, duplicateByBugId, pairs, validations = loadData(args.input)

    biggestValidation = validations[-1]
    bugReportDataset = BugReportDatabase.fromJson(args.bug_dataset)
    bugIds = list(bugIds)
    similarityListByDuplicate = []

    if args.model_type == 'tfidf':
        # Load Model
        global vectorByBug
        vectorByBug = {}
        tfIdfVectorizer = pickle.load(open(args.model, 'rb'))

        # Generate bag of words representation for each bug
        texts = [concatenateSummaryAndDescription(bugReportDataset.getBug(bugId)) for bugId in bugIds]
        vectors = tfIdfVectorizer.transform(texts)

        for idx, bugId in enumerate(bugIds):
            vectorByBug[bugId] = vectors[idx]

    else:
        # We can't import torch without allocating a GPU in Cedar cluster.
        from experiments.duplicate_bug_detection_deep_learning import generateBugEmbeddings, \
            calculateSimilarityScoresDL, \
            CosinePrediction, getDataHandlerLexiconEmb, getModel
        import torch
        import torch.nn.functional as F
        from util.torch_util import softmaxPrediction, getVariable
        from data.dataset import BugDataExtractor

        # Load Model and DataHandlers
        arguments = Obj({
            'load': args.model,
            'cuda': args.cuda,
            'summary_bidirectional': False,
            'classifier_hidden_size': 300,
            'classifier_mul_dif': True
        })

        dataHandlers, lexicons, embeddings, arguments = getDataHandlerLexiconEmb(arguments)
        encoderContainer, model = getModel(dataHandlers, lexicons, embeddings, arguments)

        encoderContainer.eval()
        model.eval()

        # Set the similarity and prediction functions
        if arguments.model == 'classification':
            similarityFunction = model[1]

            if args.cuda:
                similarityFunction.cuda()

            predictionFunction = softmaxPrediction
        elif arguments.model == 'retrieval':
            similarityFunction = F.cosine_similarity
            predictionFunction = CosinePrediction(args.retrieval_threshold, args.cuda)

        if args.cuda:
            model.cuda()
            encoderContainer.cuda()

        # Generate the embedding for each bug
        logger.info("Generating Embeddings")
        dataExtractor = BugDataExtractor(bugReportDataset, dataHandlers)
        bugEmbeddingsById = generateBugEmbeddings(bugIds, dataExtractor, encoderContainer)

    # Start to calculate all duplicate pairs recommend list
    c = time()

    logger.info("Calculating similarity scores")
    dupDictItems = duplicateByBugId.items()

    if args.model_type == 'tfidf':
        # Calculating the score for tf-idf. We had to parallel this step because the sequential version was too slow.
        import multiprocessing

        logger.info("Calculating cosine similarity of tf-idf model using %d processes" % (args.nb_processes))
        funcArgs = []

        duplicateBugs = [duplicateBug for duplicateBug, listOfDuplicates in dupDictItems]
        q = Queue()
        processes = []
        similarityScoresList = [0] * len(duplicateBugs)

        startToWrite = 0
        for idx, chunk in enumerate(chunks(duplicateBugs, args.nb_processes)):
            arr = RawArray(c_ulong, [int(bugId) for bugId in chunk])
            processes.append(multiprocessing.Process(target=parallel, args=(startToWrite, arr, q)))
            startToWrite += len(chunk)

        for p in processes:
            p.start()

        count = 0

        while True:
            try:
                id, scoreList = q.get()

                if id == -1:
                    # The process send a tuple (-1,None) when it is ending its work.
                    count += 1

                    # Break the loop when all processes were terminated
                    if count == len(processes):
                        break
                else:
                    similarityScoresList[id] = scoreList

            except Empty as e:
                pass

        logger.info(
            "Total time to calculate cosine similarity of %d duplicate bugs: %s " % (len(dupDictItems), time() - c))

    c = time()
    for i, (duplicateBug, listOfDuplicates) in enumerate(dupDictItems):
        # Calculate the similarity score of duplicate bug with each bug
        if args.model_type == 'tfidf':
            similarityScores = similarityScoresList.pop(0)
        else:
            similarityScores = calculateSimilarityScoresDL(duplicateBug, similarityFunction, bugEmbeddingsById, bugIds,
                                                           args.cuda)

        # Remove pair (duplicateBug, duplicateBug) and create tuples with bug id and its similarity score.
        bugScores = [(bugId, score) for bugId, score in zip(bugIds, similarityScores) if bugId != duplicateBug]
        # Sort  in descending order the bugs by probability of being duplicate
        similarityList = sorted(bugScores, key=lambda x: x[1], reverse=True)
        similarityListByDuplicate.append((duplicateBug, [t[0] for t in similarityList]))

        if i % 200 == 0 and i != 0:
            logger.info("Processed %d Duplicate bug of %d in %f" % (i, len(duplicateByBugId), time() - c))
            c = time()

    # For each different proportion, we calculate the recall rate and the precision, recall, accuracy
    recallKs = sorted([int(k) for k in args.recall_ratio_k])
    biggestKValue = recallKs[-1]
    total = len(duplicateByBugId)

    for validation in validations:
        logger.info("Calculating metrics to a validation with proportion: %d" % validation['k'])
        valitionBugIds = {}

        # Prepare data to prediction
        validationPairs = []
        targets = []
        bugIdsOfValidation = set()

        for pairIndex in validation['indexes']:
            bug1, bug2, label = pairs[pairIndex]

            validationPairs.append((bug1, bug2))

            valitionBugIds[bug1] = True
            valitionBugIds[bug2] = True

            bugIdsOfValidation.add(bug1)
            bugIdsOfValidation.add(bug2)

            targets.append(max(0, label))

        logger.debug("Amount of duplicate pairs: %d\tAmount of pairs: %d" % (
            np.count_nonzero(np.asarray(targets)), len(targets)))
        logger.debug("Amount of bugs: %d" % (len(bugIdsOfValidation)))

        logger.info("Predicting pair labels: %d" % validation['k'])
        if args.model_type == 'tfidf':
            predictions = predictTFIDF(validationPairs)
        else:
            predictions = predictDeepLearningModel(bugEmbeddingsById, validationPairs)

        # Calculate Recall Rate
        hitsPerRateK = [0] * len(recallKs)

        logger.info("Calculating Recall Rate")
        for duplicateBug, similarityList in similarityListByDuplicate:
            pos = biggestKValue + 1
            cur = 0
            listOfDuplicates = duplicateByBugId[duplicateBug]

            for bugId in similarityList:
                if bugId not in bugIdsOfValidation:
                    continue

                if bugId in listOfDuplicates:
                    pos = cur + 1
                    break

                cur += 1

                if cur >= biggestKValue:
                    break

            for idx, k in enumerate(recallKs):
                if k < pos:
                    continue

                hitsPerRateK[idx] += 1

        logger.info("Recall Rate Results:")
        for k, hit in zip(recallKs, hitsPerRateK):
            rate = float(hit) / total
            logger.info("\t\t k=%d: %.3f (%d/%d) " % (k, rate, hit, total))

        # Calculate Acc, precision, recall and f1
        accum = accuracy_score(targets, predictions, normalize=False)
        acc = accum / len(targets)
        prec, recall, f1, _ = precision_recall_fscore_support(targets, predictions)

        logger.info("Accuracy: %.3f (%d/%d)" % (acc * 100, accum, len(targets)))
        logger.info("Precision: {}\tRecall: {}\tF1:{}".format(list(np.around(prec * 100, decimals=3)),
                                                              list(np.around(recall * 100, decimals=3)),
                                                              list(np.around(f1 * 100, decimals=3))))

        logger.info("")
