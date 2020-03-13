'''
Calculates Recall Rate @ k
'''
import datetime
import itertools
import logging
import math
import multiprocessing
import os
import pickle
import random
from datetime import timedelta
from multiprocessing import Queue

from queue import Empty

import numpy as np
import ujson
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from data.preprocessing import concatenateSummaryAndDescription
from util.data_util import readDateFromBug, createChunks
from time import time

try:
    # Try to import torch modules. When we try to import torch library in the server without any gpu, we receive an error.
    from util.torch_util import padSequences
    import torch
    import torch.nn.functional as F
except:
    logging.getLogger().warning("It wasn't possible to load torch library")


def generateRecommendationListParalell(nProcesses, recallKs, modelAux, bugIds, duplicateBugs, bugReportDatabase,
                                       masterSetById, bugEmbeddingById, useCreationDate=True, onlyMaster=False,
                                       preselectListByBugId=None):
    def parallel(chunk, queue, index):
        logger = logging.getLogger()
        logger.info(
            "Process %s started to compute the similarity for %d duplicate bugs. Start idx: %d" % (
                os.getpid(), len(chunk), index))

        hitPerK, total = generateRecommendationList(recallKs, modelAux, list(bugIds), chunk, bugReportDatabase,
                                                    masterSetById,
                                                    useCreationDate, bugEmbeddingById, onlyMaster, preselectListByBugId)

        queue.put((hitPerK, total))

    # todo: fix this code
    q = Queue()
    processes = []
    for idx, chunk in enumerate(createChunks(duplicateBugs, nProcesses)):
        raise NotImplementedError()
        # arr = RawArray(c_ulong, [int(bugId) for bugId in chunk])
        # processes.append(multiprocessing.Process(target=parallel, args=(arr, q, idx)))

    for p in processes:
        p.start()

    count = 0
    hitsPerRateK = [0] * len(recallKs)
    total = 0

    while True:
        try:
            hits, t = q.get()

            for idx in range(len(hitsPerRateK)):
                hitsPerRateK[idx] += hits[idx]

            total += t
            count += 1

            if count == len(processes):
                break

        except Empty as e:
            pass

    return hitsPerRateK, total


def generateRecommendationList(anchorId, candidates, scorer):
    similarityScores = scorer.score(anchorId, candidates)

    # Remove pair (duplicateBug, duplicateBug) and create tuples with bug id and its similarity score.
    bugScores = [(bugId, score) for bugId, score in zip(candidates, similarityScores) if bugId != anchorId]
    # Sort  in descending order the bugs by probability of being duplicate
    sortedBySimilarity = sorted(bugScores, key=lambda x: x[1], reverse=True)

    return sortedBySimilarity


class REP_CADD_Recommender(object):

    def __init__(self, rep, rep_input_by_id, recommendation_size):
        self.rep = rep
        self.rep_input_by_id = rep_input_by_id
        self.recommendation_size = recommendation_size

    def generateRecommendationList(self, anchorId, candidates, scorer):
        anchor_input = self.rep_input_by_id[anchorId]
        similarityScores = [(cand_id, self.rep.similarity(anchor_input, self.rep_input_by_id[cand_id])) for cand_id in
                            candidates]

        sortedSimilarityScores = sorted(similarityScores, key=lambda x: x[1], reverse=True)
        cadd_candidates = [cand_id for (cand_id, score) in sortedSimilarityScores[:self.recommendation_size]]

        cadd_list = generateRecommendationList(anchorId, cadd_candidates, scorer)
        cadd_list.extend(map(lambda k: (k[0], 0), sortedSimilarityScores[self.recommendation_size:]))

        return cadd_list


class TFIDFScorer(object):

    def __init__(self, tfIdfVectorizer, bugReportDatabase):
        self.tfIdfVectorizer = tfIdfVectorizer
        self.bugReportDatabase = bugReportDatabase
        self.bugEmbeddingById = {}

    def pregenerateBugEmbedding(self, bugIds):
        # Generate bag of words representation for each bug
        texts = [concatenateSummaryAndDescription(self.bugReportDatabase.getBug(bugId)) for bugId in bugIds]
        vectors = self.tfIdfVectorizer.transform(texts)

        for idx, bugId in enumerate(bugIds):
            self.bugEmbeddingById[bugId] = vectors[idx]

    def score(self, anchorBugId, bugIds):
        batchSize = 1024
        nPairs = len(bugIds)
        nBatches = math.ceil(float(nPairs) / batchSize)

        bugEmbedding1 = self.bugEmbeddingById[anchorBugId]
        similarityScores = []
        nbDim = bugEmbedding1.shape[1]
        # todo: optimize this code
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

                bugEmbedding2 = self.bugEmbeddingById[otherBug]

                data2.extend(bugEmbedding2.data)
                indices2.extend(bugEmbedding2.indices)
                ptrs2.append(len(indices2))

            matrix1 = csr_matrix((data1, indices1, ptrs1), shape=(len(ptrs1) - 1, nbDim))
            matrix2 = csr_matrix((data2, indices2, ptrs2), shape=(len(ptrs2) - 1, nbDim))

            score = cosine_similarity(matrix1, matrix2)

            for i in range(score.shape[0]):
                similarityScores.append(score[i][i])

        return similarityScores

    def reset(self):
        pass


class SharedEncoderNNScorer(object):

    def __init__(self, preprocessorList, inputHandler, model, device, ranking_batch_size):
        self.device = device
        self.preprocessorList = preprocessorList
        self.inputHandler = inputHandler
        self.model = model
        self.bugEmbeddingById = {}
        self.ranking_batch_size = ranking_batch_size

    def pregenerateBugEmbedding(self, allBugIds):
        # Cache the bug representation of each bug
        batchSize = self.ranking_batch_size
        nIterations = int(math.ceil(len(allBugIds) / float(batchSize)))

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for it in range(nIterations):
                bugInBatch = allBugIds[it * batchSize: (it + 1) * batchSize]

                """
                We separate the information of the bugs and put each information in a different list. 
                This list represents a batch and it will be passed to a specific encoder which is responsible to 
                encode a information type in a vector.
                """
                infoBugBatch = [[] for _ in self.inputHandler]

                for bugId in bugInBatch:
                    # Preprocess raw data
                    bugInfo = self.preprocessorList.extract(bugId)

                    # Put the same information source in a specific list
                    for infoIdx, infoInput in enumerate(bugInfo):
                        infoBugBatch[infoIdx].append(infoInput)

                # Prepare data to pass it to the encoders
                model_input = []

                for inputHandler, infoBatch in zip(self.inputHandler, infoBugBatch):
                    data = inputHandler.prepare(infoBatch)
                    new_data = []
                    for d in data:
                        if d is None:
                            new_data.append(None)
                        else:
                            new_data.append(d.to(self.device))

                    model_input.append(new_data)

                encoderOutput = self.model.encode(model_input).detach().cpu().numpy()

                for idx, bugId in enumerate(bugInBatch):
                    self.bugEmbeddingById[bugId] = encoderOutput[idx]

    def score(self, anchorBugId, bugIds):
        anchorEmbedding = torch.from_numpy(self.bugEmbeddingById[anchorBugId])

        self.model.eval()
        self.model.to(self.device)

        similarityScores = []
        batchSize = self.ranking_batch_size * 2
        nPairs = len(bugIds)
        nBatches = math.ceil(float(nPairs) / batchSize)

        with torch.no_grad():
            for batchIdx in range(nBatches):
                batchStart = batchIdx * batchSize

                otherBugsBatch = torch.from_numpy(np.stack(
                    [self.bugEmbeddingById[otherBugId] for otherBugId in
                     bugIds[batchStart: batchStart + batchSize]])).to(device=self.device)
                anchorBatch = torch.as_tensor(anchorEmbedding.repeat((otherBugsBatch.shape[0], 1)), device=self.device)
                output = self.model.similarity(anchorBatch, otherBugsBatch).detach().cpu().numpy()

                # Sometimes output can be scalar (when there is only output)
                output = np.atleast_1d(output)

                for pr in output:
                    if isinstance(pr, np.float32):
                        similarityScores.append(pr)
                    else:
                        similarityScores.append(pr[-1])

        return similarityScores

    def reset(self):
        pass

    def free(self):
        pass


class DBR_CNN_Scorer(object):

    def __init__(self, categorical_prep, textual_prep, categorical_handler, textual_handler, model, device,
                 ranking_batch_size):
        self.device = device
        self.categorical_prep = categorical_prep
        self.textual_prep = textual_prep
        self.textual_handler = textual_handler
        self.categorical_handler = categorical_handler
        self.model = model
        self.bugEmbeddingById = {}
        self.ranking_batch_size = ranking_batch_size

    def pregenerateBugEmbedding(self, allBugIds):
        # Cache the bug representation of each bug
        batchSize = self.ranking_batch_size
        nIterations = int(math.ceil(len(allBugIds) / float(batchSize)))

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for it in range(nIterations):
                bugInBatch = allBugIds[it * batchSize: (it + 1) * batchSize]

                """
                We separate the information of the bugs and put each information in a different list. 
                This list represents a batch and it will be passed to a specific encoder which is responsible to 
                encode a information type in a vector.
                """
                textual_input = []

                for bugId in bugInBatch:
                    # Preprocess raw data
                    textual_input.append(self.textual_prep.extract(bugId))

                # Prepare data to pass it to the encoders
                model_input = (self.textual_handler.prepare(textual_input)[0].to(self.device),)
                encoderOutput = self.model.encode(model_input).detach().cpu().numpy()

                for idx, bugId in enumerate(bugInBatch):
                    self.bugEmbeddingById[bugId] = (self.categorical_prep.extract(bugId), encoderOutput[idx])

    def score(self, anchorBugId, bugIds):
        query_categorical, query_textual = self.bugEmbeddingById[anchorBugId]
        query_textual = torch.as_tensor(query_textual)
        query_component = np.asarray([[query_categorical[0]]], dtype="float32")
        query_priority = np.asarray([[query_categorical[1]]], dtype="float32")
        query_create_time = np.asarray([[query_categorical[2]]], dtype="float32")

        self.model.eval()
        self.model.to(self.device)

        similarityScores = []
        batchSize = self.ranking_batch_size * 2
        nPairs = len(bugIds)
        nBatches = math.ceil(float(nPairs) / batchSize)

        with torch.no_grad():
            for batchIdx in range(nBatches):
                batchStart = batchIdx * batchSize

                cand_emb = []
                cand_categorical = []
                st = time()

                for cand_id in itertools.islice(bugIds, batchStart, batchStart + batchSize):
                    cand_cat, cand_text = self.bugEmbeddingById[cand_id]

                    cand_categorical.append(cand_cat)
                    cand_emb.append(cand_text)

                cand_categorical = np.asarray(cand_categorical, dtype="float32")
                cand_categorical = (torch.from_numpy(cand_categorical[:, 0:1]).to(self.device),
                                    torch.from_numpy(cand_categorical[:, 1:2]).to(self.device),
                                    torch.from_numpy(cand_categorical[:, 2:3]).to(self.device),)

                cand_emb = torch.from_numpy(np.asarray(cand_emb)).to(device=self.device)

                query_categorical = (
                    torch.as_tensor(query_component.repeat(cand_emb.shape[0], 0), device=self.device),
                    torch.as_tensor(query_priority.repeat(cand_emb.shape[0], 0), device=self.device),
                    torch.as_tensor(query_create_time.repeat(cand_emb.shape[0], 0), device=self.device),
                )
                query_emb = torch.as_tensor(query_textual.repeat((cand_emb.shape[0], 1)), device=self.device)
                output = self.model.similarity(query_categorical, query_emb, cand_categorical,
                                               cand_emb).detach().cpu().numpy()

                # Sometimes output can be scalar (when there is only output)
                output = np.atleast_1d(output)

                for pr in output:
                    if isinstance(pr, np.float32):
                        similarityScores.append(pr)
                    else:
                        similarityScores.append(pr[-1])
        return similarityScores

    def reset(self):
        pass

    def free(self):
        pass


class RankingData(object):

    def __init__(self, preprocessingList):
        self.anchor_input = None
        self.bug_ids = None
        self.preprocessingList = preprocessingList

    def reset(self, anchor_input, bug_ids):
        self.anchor_input = anchor_input
        self.bug_ids = bug_ids

    def __len__(self):
        return len(self.bug_ids)

    def __getitem__(self, idx):
        return [self.anchor_input, self.preprocessingList.extract(self.bug_ids[idx]), 0.0]


class GeneralScorer(object):

    def __init__(self, model, preprocessingList, device, collate, batchSize=32, n_subprocess=1):
        self.device = device
        self.preprocessingList = preprocessingList
        self.model = model
        self.bugEmbeddingById = {}
        self.collate = collate
        self.batchSize = batchSize
        self.rankingdata = RankingData(preprocessingList)
        self.data_loader = DataLoader(self.rankingdata,
                                      batch_size=batchSize, shuffle=False, collate_fn=collate.collate,
                                      num_workers=n_subprocess, )

    def pregenerateBugEmbedding(self, allBugIds):
        pass

    def score(self, candidate_id, bug_ids):
        self.model.eval()
        self.model.to(self.device)

        similarityScores = []
        candidate_bug = self.preprocessingList.extract(candidate_id)
        self.rankingdata.reset(candidate_bug, bug_ids)

        with torch.no_grad():
            for batch in self.data_loader:
                # Transfer data to GPU
                x, y = self.collate.to(batch, self.device)

                output = self.model(*x).detach().cpu().numpy()

                # Sometimes output can be scalar (when there is only output)
                output = np.atleast_1d(output)

                for pr in output:
                    if isinstance(pr, np.float32):
                        similarityScores.append(pr)
                    else:
                        similarityScores.append(pr[-1])

        return similarityScores

    def reset(self):
        pass

    def free(self):
        pass


# Implement the three methods to calculate the recall rate

class SunRanking(object):

    def __init__(self, bugReportDatabase, dataset, window):
        self.bugReportDatabase = bugReportDatabase
        self.masterIdByBugId = self.bugReportDatabase.getMasterIdByBugId()
        self.duplicateBugs = dataset.duplicateIds
        self.candidates = []
        self.window = int(window) if window is not None else 0
        self.latestDateByMasterSetId = {}
        self.logger = logging.getLogger()

        # Get oldest and newest duplicate bug report in dataset
        oldestDuplicateBug = (
            self.duplicateBugs[0], readDateFromBug(self.bugReportDatabase.getBug(self.duplicateBugs[0])))

        for dupId in self.duplicateBugs:
            dup = self.bugReportDatabase.getBug(dupId)
            creationDate = readDateFromBug(dup)

            if oldestDuplicateBug[1] < creationDate:
                oldestDuplicateBug = (dupId, creationDate)

        # Keep only master that are able to be candidate
        for bug in self.bugReportDatabase.bugList:
            bugCreationDate = readDateFromBug(bug)
            bugId = bug['bug_id']

            # Remove bugs that their creation time is bigger than oldest duplicate bug
            if bugCreationDate > oldestDuplicateBug[1] or (
                    bugCreationDate == oldestDuplicateBug[1] and bug['bug_id'] > oldestDuplicateBug[0]):
                continue

            self.candidates.append((bugId, bugCreationDate.timestamp()))

        # Keep the timestamp of all reports in each master set
        for masterId, masterSet in self.bugReportDatabase.getMasterSetById(
                map(lambda c: c[0], self.candidates)).items():
            ts_list = []

            for bugId in masterSet:
                bugCreationDate = readDateFromBug(self.bugReportDatabase.getBug(bugId))

                ts_list.append((int(bugId), bugCreationDate.timestamp()))

            self.latestDateByMasterSetId[masterId] = ts_list

        # Set all bugs that are going to be used by our models.
        self.allBugs = [bugId for bugId, bugCreationDate in self.candidates]
        self.allBugs.extend(self.duplicateBugs)

    def getDuplicateBugs(self):
        return self.duplicateBugs

    def getAllBugs(self):
        return self.allBugs

    def getCandidateList(self, anchorId):
        candidates = []
        anchor = self.bugReportDatabase.getBug(anchorId)
        anchorCreationDate = readDateFromBug(anchor)
        anchorMasterId = self.masterIdByBugId[anchorId]
        nDupBugs = 0
        anchorTimestamp = anchorCreationDate.timestamp()
        anchorDayTimestamp = int(anchorTimestamp / (24 * 60 * 60))

        nSkipped = 0
        window_record = [] if self.logger.isEnabledFor(logging.DEBUG) else None
        anchorIdInt = int(anchorId)

        for bugId, bugCreationDate in self.candidates:
            bugIdInt = int(bugId)

            # Ignore reports that were created after the anchor report
            if bugCreationDate > anchorTimestamp or (
                    bugCreationDate == anchorTimestamp and bugIdInt > anchorIdInt):
                continue

            # Check if the same report
            if bugId == anchorId:
                continue

            if bugIdInt > anchorIdInt:
                self.logger.warning(
                    "Candidate - consider a report which its id {} is bigger than duplicate {}".format(bugId, anchorId))

            masterId = self.masterIdByBugId[bugId]

            # Group all the duplicate and master in one unique set. Creation date of newest report is used to filter the bugs
            tsMasterSet = self.latestDateByMasterSetId.get(masterId)

            if tsMasterSet:
                max = -1
                newest_report = None

                for candNewestId, ts in self.latestDateByMasterSetId[masterId]:
                    # Ignore reports that were created after the anchor or the ones that have the same ts and bigger id
                    if ts > anchorTimestamp or (ts == anchorTimestamp and candNewestId >= anchorIdInt):
                        continue

                    if candNewestId >= anchorIdInt:
                        self.logger.warning(
                            "Window filtering - consider a report which its id {} is bigger than duplicate {}".format(
                                candNewestId,
                                anchorIdInt))

                    # Get newest ones
                    if max < ts:
                        max = ts
                        newest_report = candNewestId

                # Transform to day timestamp
                bug_timestamp = int(max / (24 * 60 * 60))
            else:
                # Transform to day timestamp
                bug_timestamp = int(bugCreationDate / (24 * 60 * 60))
                newest_report = bugId

            # Is it in the window?
            if 0 < self.window < (anchorDayTimestamp - bug_timestamp):
                nSkipped += 1
                continue

            # Count number of duplicate bug reports
            if anchorMasterId == masterId:
                nDupBugs += 1

            # It is a candidate
            candidates.append(bugId)
            if window_record is not None:
                window_record.append((bugId, newest_report, bug_timestamp))

        self.logger.debug(
            "Query {} ({}) - window {} - number of reports skipped: {}".format(anchorId, anchorDayTimestamp,
                                                                               self.window, nSkipped))

        if window_record is not None:
            self.logger.debug("{}".format(window_record))

        if nDupBugs == 0:
            return []

        return candidates


class DeshmukhRanking(object):

    def __init__(self, bugReportDatabase, dataset):
        self.bugReportDatabase = bugReportDatabase
        self.masterIdByBugId = self.bugReportDatabase.getMasterIdByBugId()
        self.duplicateBugs = dataset.duplicateIds
        self.allBugs = dataset.bugIds

    def getAllBugs(self):
        return self.allBugs

    def getDuplicateBugs(self):
        return self.duplicateBugs

    def getCandidateList(self, anchorId):
        return [bugId for bugId in self.allBugs if anchorId != bugId]


class PreselectListRanking(object):

    def __init__(self, filePath, nBugToSample=0):
        self.allBugIds, self.listByBugId = pickle.load(open(filePath, 'rb'))
        self.duplicateBugs = list(self.listByBugId.keys())
        self.nBugToSample = len(self.duplicateBugs) if nBugToSample is None or nBugToSample <= 0 else nBugToSample

    def getDuplicateBugs(self):
        if self.nBugToSample == len(self.duplicateBugs):
            return self.duplicateBugs

        return random.sample(self.duplicateBugs, self.nBugToSample)

    def getCandidateList(self, anchorId):
        return self.listByBugId[anchorId]

    def getAllBugs(self):
        return self.allBugIds


class RecallRate(object):

    def __init__(self, bugReportDatabase, k=None, groupByMaster=True):
        self.masterSetById = bugReportDatabase.getMasterSetById()
        self.masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

        if k is None:
            k = list(range(1, 21))

        self.k = sorted(k)

        self.hitsPerK = dict((k, 0) for k in self.k)
        self.nDuplicate = 0
        self.logger = logging.getLogger()

        self.groupByMaster = groupByMaster

    def reset(self):
        self.hitsPerK = dict((k, 0) for k in self.k)
        self.nDuplicate = 0

    def update(self, anchorId, recommendationList):
        mastersetId = self.masterIdByBugId[anchorId]
        masterSet = self.masterSetById[mastersetId]
        # biggestKValue = self.k[-1]

        # pos = biggestKValue + 1
        pos = math.inf
        correct_cand = None

        if len(recommendationList) == 0:
            self.logger.warning("Recommendation list of {} is empty. Consider it as miss.".format(anchorId))
        else:
            seenMasters = set()

            for bugId, p in recommendationList:
                mastersetId = self.masterIdByBugId[bugId]

                if self.groupByMaster:
                    if mastersetId in seenMasters:
                        continue

                    seenMasters.add(mastersetId)
                else:
                    seenMasters.add(bugId)

                # if len(seenMasters) == pos:
                #     break

                if bugId in masterSet:
                    pos = len(seenMasters)
                    correct_cand = bugId
                    break

        # If one of k duplicate bugs is in the list of duplicates, so we count as hit. We calculate the hit for each different k
        for idx, k in enumerate(self.k):
            if k < pos:
                continue

            self.hitsPerK[k] += 1

        self.nDuplicate += 1

        return pos, correct_cand

    def compute(self):
        recallRate = {}
        for k, hit in self.hitsPerK.items():
            rate = float(hit) / self.nDuplicate
            recallRate[k] = rate

        return recallRate


class RankingResultFile(object):

    def __init__(self, filePath, bugReportDatabase):
        filename, ext = os.path.splitext(filePath)
        self.filepath = "{}_{}{}".format(filename, int(random.randint(0, 10000000)), ext)
        self.file = open(self.filepath, 'w')
        self.logger = logging.getLogger(__name__)

        self.logger.info({"result_path":self.filepath})

    def update(self, anchorId, recommendationList, pos, correct_cand):
        self.file.write(anchorId)

        for cand_id, score in recommendationList:
            self.file.write(" ")
            self.file.write(cand_id)
            self.file.write("|")
            self.file.write(str(round(score, 3)))

        self.file.write("\n")
        self.file.flush()
