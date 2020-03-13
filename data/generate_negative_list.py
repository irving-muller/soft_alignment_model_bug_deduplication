import argparse
import heapq
import logging
import multiprocessing
import os
import pickle
from ctypes import c_ulong
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from queue import Empty
from time import time

import sklearn

from classical_approach.read_data import read_weights, read_dbrd_file
from classical_approach.bm25f import SUN_REPORT_ID_INDEX, SUN_REPORT_DID_INDEX
from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from data.preprocessing import concatenateSummaryAndDescription
from util.data_util import createChunks

import h5py


def iterateListSunModel(query, reports, model):
    for report in reports:
        if (len(report[SUN_REPORT_DID_INDEX]) != 0 and report[SUN_REPORT_DID_INDEX] == query[SUN_REPORT_DID_INDEX]) \
                or query == report:
            continue

        yield (model.similarity(query, report), report[SUN_REPORT_ID_INDEX])


def generateNegativeListSunModel(listSize, bugDataFile, training_dataset, model, nProcesses):
    logger = logging.getLogger(__name__)

    max_bug_id = max(map(lambda bug_id: int(bug_id), training_dataset.bugIds))
    reports, max_token_id = read_dbrd_file(bugDataFile, max_bug_id)

    logger.info("Calculating IDF and average lengths and reading dataset")
    model.fit_transform(reports, max_token_id, True)

    def parallel(chunk, queue, index):
        logger = logging.getLogger()
        logger.info(
            "Process %s started to compute the similarity for %d duplicate bugs. Start idx: %d" % (
                os.getpid(), len(chunk), index))

        start = time()
        for idx, bugIdx in enumerate(chunk):
            query = reports[bugIdx]
            nMostSimilar = heapq.nlargest(listSize, iterateListSunModel(query, reports, model))

            output = (query[SUN_REPORT_ID_INDEX], nMostSimilar)

            if (idx + 1) % 100 == 0:
                t = time() - start
                logger.info("%s computed similarity list for %d of %d in %f seconds" % (
                    os.getpid(), idx + 1, len(chunk), t))
                start = time()

            queue.put(output)

        queue.put([])

        logger.info("Process %s has finished" % (os.getpid()))

    bugIdxs = list(range(len(reports)))

    q = Queue()
    processes = []

    for idx, chunk in enumerate(createChunks(bugIdxs, nProcesses)):
        arr = RawArray(c_ulong, chunk)
        processes.append(multiprocessing.Process(target=parallel, args=(arr, q, idx)))

    for p in processes:
        p.start()

    count = 0
    ret_report_ids = []
    recommendation_matrix = []
    best_matrix = []
    worst_matrix = []
    n = 30

    while True:
        try:
            out = q.get()

            if len(out) == 0:
                count += 1
                logger.info("One process was ended up! Count: %d/%d" % (count, len(processes)))
            else:
                bugId, simList = out

                recommendationList = []
                bestlist = []
                worstlist = []

                for idx, (cosine, negBugId) in enumerate(simList):
                    recommendationList.append(int(negBugId))

                    if idx < n:
                        bestlist.append(cosine)

                    if idx > len(simList) - n:
                        worstlist.append(cosine)

                ret_report_ids.append(int(bugId))
                recommendation_matrix.append(recommendationList)
                best_matrix.append(bestlist)
                worst_matrix.append(worstlist)

            nProcessedBugs = len(ret_report_ids)

            if nProcessedBugs % 100 == 0:
                logger.info("Main Thread: Processed %d " % (nProcessedBugs))

            if nProcessedBugs == len(bugIdxs):
                logger.info("It is over!")
                break

            if count == len(processes):
                break
        except Empty as e:
            pass

    return ret_report_ids, recommendation_matrix, best_matrix, worst_matrix


def iterateSimList(bugIds, similarityMatrix, masterId, masterBugIdByBugId, product, bugReportDatabase):
    for otherBugId, cosineSim in zip(bugIds, similarityMatrix):
        if masterBugIdByBugId[otherBugId] == masterId:
            continue

        if product and product != bugReportDatabase.getBug(otherBugId)['product']:
            continue

        yield (cosineSim[0], otherBugId)


def generateNegativeListSparseVector(listSize, bugReportDatabase, bugIds, vectorizerClass, masterBugIdByBugId,
                                     normalize, nProcesses, sameProd=False):
    bugTexts = [concatenateSummaryAndDescription(bugReportDatabase.getBug(bugId)) for bugId in bugIds]
    logger = logging.getLogger(__name__)

    logger.info("Transforming text to vector")
    vectors = vectorizerClass.transform(bugTexts)

    if normalize:
        logger.info("Normalizing vectors to length 1")
        matrixRep = sklearn.preprocessing.data.normalize(vectors)
    else:
        matrixRep = vectors

    similarityIterByBugId = {}

    # Cache the similarity list of the bugs
    logger.info("Starting to cache the similarity list")

    bugToCreateList = []
    bugPosition = {id: idx for idx, id in enumerate(bugIds)}

    for master in bugReportDatabase.getMasterSetById(bugIds).values():
        for bugId in master:
            if bugPosition.get(bugId, -1) != -1:
                bugToCreateList.append(bugId)

    def parallel(chunk, queue, index):
        logger = logging.getLogger()
        logger.info(
            "Process %s started to compute the similarity for %d duplicate bugs. Start idx: %d" % (
                os.getpid(), len(chunk), index))

        start = time()
        for idx, bugIdx in enumerate(chunk):
            bugId = bugToCreateList[bugIdx]
            position = bugPosition[bugId]
            bugRep = matrixRep[position]
            product = bugReportDatabase.getBug(bugId)['product'] if sameProd else None

            similarityMatrix = matrixRep.dot(bugRep.T).toarray()

            masterId = masterBugIdByBugId.get(bugId)
            nMostSimilar = heapq.nlargest(listSize,
                                          iterateSimList(bugIds, similarityMatrix, masterId, masterBugIdByBugId,
                                                         product, bugReportDatabase))

            output = (bugId, nMostSimilar)

            if (idx + 1) % 100 == 0:
                t = time() - start
                logger.info("%s computed similarity list for %d of %d in %f seconds" % (
                    os.getpid(), idx + 1, len(chunk), t))
                start = time()

            queue.put(output)

        queue.put([])

        logger.info("Process %s has finished" % (os.getpid()))

    bugIdxs = list(range(len(bugToCreateList)))

    q = Queue()
    processes = []

    for idx, chunk in enumerate(createChunks(bugIdxs, nProcesses)):
        arr = RawArray(c_ulong, chunk)
        processes.append(multiprocessing.Process(target=parallel, args=(arr, q, idx)))

    for p in processes:
        p.start()

    count = 0

    while True:
        try:
            out = q.get()

            if len(out) == 0:
                count += 1
                logger.info("One process was ended up! Count: %d/%d" % (count, len(processes)))
            else:
                bugId, simList = out
                similarityIterByBugId[bugId] = ([int(negBugId) for cosine, negBugId in simList],
                                                [cosine for cosine, negBugId in simList[:30]],
                                                [cosine for cosine, negBugId in simList[-30:]])

            nProcessedBugs = len(similarityIterByBugId)

            if nProcessedBugs % 100 == 0:
                logger.info("Main Thread: Processed %d " % (nProcessedBugs))

            if nProcessedBugs == len(bugIdxs):
                logger.info("It is over!")
                break

            if count == len(processes):
                break
        except Empty as e:
            pass

    return similarityIterByBugId


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bug_data', required=True, help="")
    parser.add_argument('--dataset', required=True, help="")
    parser.add_argument('--list_size', required=True, type=int, help="")
    parser.add_argument('--type', required=True, help="")
    parser.add_argument('--save', help="")
    parser.add_argument('--model', help="")
    parser.add_argument('--nproc', type=int, default=6, help="")
    parser.add_argument('--same_prod', action='store_true', help="")

    logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)

    bugDataset = BugDataset(args.dataset)

    bugIds = bugDataset.bugIds
    duplicateBugs = bugDataset.duplicateIds

    if args.type in set(['tfidf', 'binary']):
        # Insert imports to load TfIdfVectorizer class
        from data.bug_dataset import BugDataset

        bugReportDatabase = BugReportDatabase.fromJson(args.bug_data)
        masterBugIdByBugId = bugReportDatabase.getMasterIdByBugId()

        vectorizer = pickle.load(open(args.model, 'rb'))
        normalize = True if args.type == 'tfidf' else False

        negativeSimMatrix = generateNegativeListSparseVector(args.list_size, bugReportDatabase, bugIds, vectorizer,
                                                             masterBugIdByBugId, normalize, args.nproc, args.same_prod)

        logger.info("Saving")
        pickle.dump(negativeSimMatrix, open(args.save, 'wb'))
    elif args.type in ['bm25f_ext', 'rep']:
        # Read the file with the weights
        sun_model = read_weights(args.model)
        ret_report_ids, recommendationMatrix, bestMatrix, worstMatrix = generateNegativeListSunModel(args.list_size,
                                                                                              args.bug_data, bugDataset,
                                                                                              sun_model, args.nproc)

        f = h5py.File(args.save, 'w')

        f.create_dataset("report_ids", data=ret_report_ids)
        f.create_dataset("recommendation", data=recommendationMatrix)
        f.create_dataset("best", data=bestMatrix)
        f.create_dataset("worst", data=worstMatrix)

        f.close()

    else:
        raise Exception('%s is not available option' % args.type)
