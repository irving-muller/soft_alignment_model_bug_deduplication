import logging
import math
import random
from math import ceil

import h5py
import numpy as np
import torch

from classical_approach.bm25f import SUN_REPORT_ID_INDEX


def sample_negative_report(masterIdByBugId, bugId, bugIdList):
    masterId = masterIdByBugId[bugId]
    current = bugIdList[random.randint(0, len(bugIdList) - 1)]
    currentMasterId = masterIdByBugId[current]

    if masterId == currentMasterId:
        return None

    return current


class BasicGenerator(object):

    def __init__(self, bugIdList, randomAnchor):
        self.possibleAnchors = None
        self.bugIdList = bugIdList
        self.randomAnchor = randomAnchor

    def setPossibleAnchors(self, pairsWithId):
        if self.randomAnchor:
            self.possibleAnchors = self.bugIdList
        else:
            if not self.possibleAnchors:
                possibleAnchors = set()

                for anchorId, candId, label in pairsWithId:
                    if label > 0:
                        possibleAnchors.add(anchorId)
                        possibleAnchors.add(candId)

                self.possibleAnchors = list(possibleAnchors)


class RandomGenerator(BasicGenerator):

    def __init__(self, preprocessor, collate, rate, bugIdList, masterIdByBugId, randomAnchor=True):
        super(RandomGenerator, self).__init__(bugIdList, randomAnchor)
        self.logger = logging.getLogger(__name__)
        self.preprocessor = preprocessor
        self.collate = collate
        self.rate = rate
        self.masterIdByBugId = masterIdByBugId

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        negatives = []

        self.setPossibleAnchors(pairsWithId)

        for posPair in range(len(posPairs)):
            for _ in range(self.rate):
                anchorId = self.possibleAnchors[random.randint(0, len(self.possibleAnchors) - 1)]
                anchorIn = self.preprocessor.extract(anchorId)
                negId = self.sampleNegativeExample(anchorId)

                negatives.append((anchorIn, self.preprocessor.extract(negId), 0))

        return posPairs, negatives

    def generateTriplets(self, model, tripletLoss, posPairs):
        """
        Given a positive pair, the class generates the new examples.
        :param model:
        :param tripletLoss:
        :param posPairs:
        :return:
        """
        triplets = []

        for (anchor, pos) in posPairs:
            anchorId, anchorIn = anchor
            posId, posIn = pos

            for _ in range(self.rate):
                negId = self.sampleNegativeExample(anchorId)

                triplets.append((anchorIn, posIn, self.preprocessor.extract(negId)))

        return triplets

    def sampleAnchor(self, k):
        return [self.possibleAnchors[random.randint(0, len(self.possibleAnchors) - 1)] for _ in range(k)]

    def sampleNegativeExample(self, bugId):
        masterId = self.masterIdByBugId[bugId]
        currentMasterId = masterId
        current = None

        while masterId == currentMasterId:
            current = self.bugIdList[random.randint(0, len(self.bugIdList) - 1)]
            currentMasterId = self.masterIdByBugId[current]

        return current

class KRandomGenerator(RandomGenerator):
    """
    Sample K negative candidates for each anchor and use the r candidates with the most higher loss value.

    Improved Representation Learning for Question Answer Matching: Ming Tan, Cicero dos Santos, Bing Xiang & Bowen Zhou
    """

    def __init__(self, preprocessor, collate, rate, bugIdList, masterIdByBugId, k, device, randomAnchor=True):
        """
        """
        super(KRandomGenerator, self).__init__(preprocessor, collate, rate, bugIdList, masterIdByBugId,
                                               randomAnchor)
        self.k = k
        self.device = device

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        negatives = []

        self.setPossibleAnchors(pairsWithId)

        for _ in range(self.rate):
            anchors = self.sampleAnchor(len(posPairs))
            self.generateExamples(negatives, model, loss, anchors, 1)

        return posPairs, negatives

    def generateTriplets(self, model, tripletLoss, posPairs):
        """
        Given a positive pair, the class generates the new examples.
        :param model:
        :param loss:
        :param posPairs:
        :return:
        """
        triplets = []
        self.generateExamples(triplets, model, tripletLoss, posPairs, self.rate)

        return triplets

    def generateExamples(self, negativeList, model, lossFun, anchors, nNegativePerAnchor):
        batchSize = 128
        nAnchorPerTime = int(float(batchSize) / self.k)
        currentIdx = 0
        nmOfZeroLoss = 0

        if nAnchorPerTime == 0:
            nAnchorPerTime = 1

        model.eval()
        with torch.no_grad():
            while currentIdx < len(anchors):
                batch = []
                nAnchor = 0

                for anchor in anchors[currentIdx:currentIdx + nAnchorPerTime]:
                    if isinstance(anchor, (list, tuple)):
                        anchorId, anchorEmb = anchor[0]
                        posId, posEmb = anchor[1]

                        for _ in range(self.k):
                            negEmb = self.preprocessor.extract(self.sampleNegativeExample(anchorId))
                            batch.append((anchorEmb, posEmb, negEmb))
                    else:
                        anchorId = anchor
                        anchorEmb = self.preprocessor.extract(anchorId)

                        for _ in range(self.k):
                            negEmb = self.preprocessor.extract(self.sampleNegativeExample(anchorId))
                            batch.append((anchorEmb, negEmb, 0.0))

                    nAnchor += 1
                currentIdx += nAnchor

                x = self.collate.collate(batch)
                input, target = self.collate.to(x, self.device)

                output = model(*input)

                lossValue = lossFun(output, target)
                lossValue = lossValue.view(nAnchor, self.k)

                lossValues, idxs = torch.sort(lossValue, dim=1, descending=True)

                lossValues = lossValues.data.cpu().numpy()
                idxs = idxs.data.cpu().numpy()

                for anchorIdx, negIdxs in enumerate(idxs):
                    for colIdx, negIdx in enumerate(negIdxs[:nNegativePerAnchor]):
                        if np.around(lossValues[anchorIdx][colIdx], 6) == 0.0:
                            nmOfZeroLoss += 1

                        negativeList.append(batch[anchorIdx * self.k + negIdx])

        self.logger.info("==> KRandom: number of triplets with loss zero=%d / %d" % (nmOfZeroLoss, len(anchors)))

        return negativeList


class NonNegativeRandomGenerator(RandomGenerator):
    """
    Generate the negative pairs which the nn loss is bigger than alpha.
    """

    def __init__(self, preprocessor, collate, rate, bugIdList, masterIdByBugId, nTries, device, randomAnchor=True,
                 silence=False, decimals=3):
        """
        """
        super(NonNegativeRandomGenerator, self).__init__(preprocessor, collate, rate, bugIdList, masterIdByBugId,
                                                         randomAnchor)
        self.nTries = nTries
        self.silence = silence
        self.decimals = decimals
        self.device = device

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        negativePairs = []

        self.setPossibleAnchors(pairsWithId)

        for i in range(self.nTries):
            if not self.silence:
                self.logger.info("==> Try {}".format(i + 1))

            for _ in range(self.rate):
                anchors = self.sampleAnchor(len(posPairs))

                for negPair in self.generateExamples(model, loss, anchors):
                    negativePairs.append(negPair)

                    if len(negativePairs) == len(posPairs) * self.rate:
                        return posPairs, negativePairs

            if not self.silence:
                self.logger.info("==> Try {} - we still have to generate {} good pairs.".format(
                    i + 1, len(posPairs) * self.rate - len(negativePairs)))

        missing_negative = len(posPairs) * self.rate - len(negativePairs)

        if missing_negative > 0:
            self.logger.info(
                "We generated a number of negative pairs (%d) that was insufficient to maintain the same rate." % (
                    len(negativePairs)))
            nPosPairs = int(len(negativePairs) / float(self.rate))
            self.logger.info("Randomly select %d positive pairs to maintain the rate." % (nPosPairs))
            posPairs = random.sample(posPairs, nPosPairs)
            # self.logger.info("Randomly select {} negative pairs to maintain the rate ({} * {}).".format(
            #     missing_negative, len(posPairs), self.rate))
            #
            # for anchorId in self.sampleAnchor(missing_negative):
            #     anchorIn = self.preprocessor.extract(anchorId)
            #     negId = self.sampleNegativeExample(anchorId)
            #
            #     negativePairs.append((anchorIn, self.preprocessor.extract(negId), 0))

        return posPairs, negativePairs

    def generateTriplets(self, model, tripletLoss, posPairs):
        """
        Given a positive pair, the class generates the new examples.
        :param model:
        :param loss:
        :param posPairs:
        :return:
        """
        triplets = []

        for i in range(self.nTries):
            self.logger.info("==> Try {}".format(i + 1))
            for _ in range(self.rate):
                for newTriplet in self.generateExamples(model, tripletLoss, posPairs):
                    triplets.append(newTriplet)

                    if len(triplets) == len(posPairs) * self.rate:
                        return triplets

            self.logger.info("==> Try {} - we still have to generate {} good pairs.".format(
                i + 1, len(posPairs) * self.rate - len(triplets)))

        if len(triplets) < len(posPairs) * self.rate:
            self.logger.info(
                "We generated a number of new triplets (%d) that was insufficient to maintain the same rate." % (
                    len(triplets)))

        return triplets

    def generateExamples(self, model, lossFun, anchors):
        batchSize = 128
        negatives = []
        nIteration = math.ceil(len(anchors) / float(batchSize))
        decimals = self.decimals

        model.eval()

        with torch.no_grad():
            for it in range(nIteration):
                batch = []
                bugIds = []

                for anchor in anchors[it * batchSize: (it + 1) * batchSize]:
                    if isinstance(anchor, (list, tuple)):
                        anchorId, anchorEmb = anchor[0]
                        posId, posEmb = anchor[1]

                        negId = self.sampleNegativeExample(anchorId)
                        negEmb = self.preprocessor.extract(negId)

                        bugIds.append((anchorId, posId, negId))
                        batch.append((anchorEmb, posEmb, negEmb))
                    else:
                        anchorId = anchor
                        anchorEmb = self.preprocessor.extract(anchorId)
                        negId = self.sampleNegativeExample(anchorId)
                        negEmb = self.preprocessor.extract(negId)

                        bugIds.append((anchorId, negId))
                        batch.append((anchorEmb, negEmb, 0.0))

                x = self.collate.collate(batch)
                input, target = self.collate.to(x, self.device)
                output = model(*input)

                lossValue = lossFun(output, target).flatten()
                lossValues, idxs = torch.sort(lossValue, descending=True)

                lossValues = lossValues.data.cpu().numpy().flatten()
                idxs = idxs.data.cpu().numpy().flatten()

                for idx, lossValue in zip(idxs, lossValues):
                    if np.around(lossValue, decimals=decimals) > 0.0:
                        yield batch[idx]


class ProductComponentRandomGen(NonNegativeRandomGenerator):

    def __init__(self, report_db, preprocessor, collate, rate, bugIdList, masterIdByBugId, nTries, device,
                 randomAnchor=True, silence=False, decimals=3):
        super(NonNegativeRandomGenerator, self).__init__(preprocessor, collate, rate, bugIdList, masterIdByBugId,
                                                         randomAnchor)
        self.nTries = nTries
        self.silence = silence
        self.decimals = decimals
        self.device = device
        self.report_db = report_db

        self.bug_ids_by_product = {}
        self.bug_ids_by_product_comp = {}

        for bug_id in bugIdList:
            bug = self.report_db.getBug(bug_id)

            product = bug['product'].lower()
            component = bug['component'].lower()

            self.bug_ids_by_product.setdefault(product, []).append(bug_id)
            self.bug_ids_by_product_comp.setdefault(product + " " + component, []).append(bug_id)

    def sampleNegativeExample(self, bug_id):
        candidate_id = None

        while candidate_id is None:
            p = random.random()

            if p < 0.33:
                eligible_candidates = self.bugIdList
            elif 0.33 <= p < 0.66:
                bug = self.report_db.getBug(bug_id)
                eligible_candidates = self.bug_ids_by_product[bug['product'].lower()]
            elif 0.66 <= p:
                bug = self.report_db.getBug(bug_id)

                product = bug['product'].lower()
                component = bug['component'].lower()

                eligible_candidates = self.bug_ids_by_product_comp[product + " " + component]

            candidate_id = sample_negative_report(self.masterIdByBugId, bug_id, eligible_candidates)


        return candidate_id


class MiscNonZeroRandomGen(NonNegativeRandomGenerator):
    """
    Half of the negatives pairs are randomly created while the other half has negative pairs which one bug is in positive pair list
    """

    def __init__(self, preprocessor, collate, rate, bugIdList, duplicateBugs, masterIdByBugId, nTries, device,
                 randomAnchor=True):
        """
        """
        super(MiscNonZeroRandomGen, self).__init__(preprocessor, collate, rate, bugIdList, masterIdByBugId, nTries,
                                                   device, randomAnchor)

        allBugs = set(bugIdList)
        bugInPosPairs = set(duplicateBugs)

        for b1 in duplicateBugs:
            master = masterIdByBugId[b1]

            if master not in bugInPosPairs and master in allBugs:
                bugInPosPairs.add(master)

        self.bugInPosPairs = list(bugInPosPairs)

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        negativePairs = []

        self.setPossibleAnchors(pairsWithId)

        nNegPairs = len(posPairs) * self.rate
        nRandomPairs = int(nNegPairs / 2)
        self.logger.info("Randomly generating {} negatives pairs".format(nRandomPairs))

        for i in range(self.nTries):
            self.logger.info("==> Try {}".format(i + 1))
            for _ in range(self.rate):
                anchors = self.sampleAnchor(nRandomPairs)

                for negPair in self.generateExamples(model, loss, anchors):
                    negativePairs.append(negPair)

                    if len(negativePairs) == nRandomPairs:
                        break

            if len(negativePairs) == nRandomPairs:
                break

            self.logger.info("==> Try {} - we still have to generate {} good random pairs.".format(
                i + 1, len(posPairs) * self.rate - len(negativePairs)))

        remaining = nNegPairs - len(negativePairs)
        self.logger.info("Generating {} negatives pairs which one bug is duplicate".format(remaining))

        for i in range(self.nTries):
            random.shuffle(self.bugInPosPairs)

            self.logger.info("==> Try {}".format(i + 1))
            for _ in range(self.rate):
                for negPair in self.generateExamples(model, loss, self.bugInPosPairs):
                    negativePairs.append(negPair)

                    if len(negativePairs) == nNegPairs:
                        return posPairs, negativePairs

            self.logger.info("==> Try {} - we still have to generate {} good random pairs.".format(
                i + 1, len(posPairs) * self.rate - len(negativePairs)))

        if len(negativePairs) < len(posPairs) * self.rate:
            self.logger.info(
                "We generated a number of negative pairs (%d) that was insufficient to maintain the same rate." % (
                    len(negativePairs)))
            nPosPairs = int(len(negativePairs) / float(self.rate))
            self.logger.info("Randomly select %d positive pairs to maintain the rate." % (nPosPairs))
            posPairs = random.sample(posPairs, nPosPairs)

        return posPairs, negativePairs


class REPGenerator(BasicGenerator):

    def __init__(self, rep, rep_input, recommendation_size, preprocessor, bugIdList, masterIdByBugId, rate,
                 randomAnchor=True):
        super(REPGenerator, self).__init__(None, randomAnchor)
        self.logger = logging.getLogger(__name__)
        self.rep = rep
        self.bugIdList = bugIdList
        self.rate = rate
        self.preprocessor = preprocessor

        if isinstance(rep_input, list):
            rep_input_by_id = {}

            for inp in rep_input:
                rep_input_by_id[inp[SUN_REPORT_ID_INDEX]] = inp
        else:
            rep_input_by_id = rep_input

        self.rep_input_by_id = rep_input_by_id
        self.recommendation_size = recommendation_size
        self.masterIdByBugId = masterIdByBugId

    def sampleNegativeExample(self, bugId, n):
        negative_examples = []

        while len(negative_examples) < n:
            masterId = self.masterIdByBugId[bugId]
            currentMasterId = masterId
            current = None

            while masterId == currentMasterId:
                current = self.bugIdList[random.randint(0, len(self.bugIdList) - 1)]
                currentMasterId = self.masterIdByBugId[current]

            negative_examples.append(current)

        return negative_examples

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        negativePairs = []

        self.setPossibleAnchors(pairsWithId)

        for _ in range(self.rate):
            for idx in range(len(posPairs)):
                anchorId = self.possibleAnchors[random.randint(0, len(self.possibleAnchors) - 1)]
                anchor_input = self.rep_input_by_id[anchorId]
                negative_examples = self.sampleNegativeExample(anchorId, self.recommendation_size)
                similarity_scores = [
                    (self.rep.similarity(anchor_input, self.rep_input_by_id[cand_id]), cand_id)
                    for cand_id in negative_examples]

                sorted_similarity_scores = sorted(similarity_scores, reverse=True)

                for score, cand_id in sorted_similarity_scores[:self.rate]:
                    negativePairs.append(
                        (self.preprocessor.extract(str(anchorId)), self.preprocessor.extract(cand_id), 0))

        self.logger.info(
            "It was generated {} negative pairs. Positive pairs={}".format(len(negativePairs), len(posPairs)))

        return posPairs, negativePairs

    def generateTriplets(self, model, tripletLoss, posPairs):
        """
        Given a positive pair, the class generates the new examples.
        :param model:
        :param loss:
        :param posPairs:
        :return:
        """
        triplets = []

        for anchor in posPairs:
            anchorId, anchorEmb = anchor[0]
            posId, posEmb = anchor[1]

            negative_examples = self.sampleNegativeExample(anchorId, self.recommendation_size)
            similarity_scores = [
                (self.rep.similarity(self.rep_input_by_id[anchorId], self.rep_input_by_id[cand_id]), cand_id)
                for cand_id in negative_examples]

            for score, cand_id in similarity_scores[:self.rate]:
                triplets.append(anchorEmb, posEmb, self.preprocessor.extract(cand_id))

        self.logger.info("It was generated {} triplets".format(len(triplets)))

        return triplets


class PreSelectedGenerator(BasicGenerator):

    def __init__(self, negativeListFile, preprocessor, rate, masterIdByBugId, maxRecommendation=None,
                 randomAnchor=True):
        super(PreSelectedGenerator, self).__init__(None, randomAnchor)
        self.logger = logging.getLogger(__name__)
        self.rate = rate
        self.logger = logging.getLogger(__name__)
        self.preprocessor = preprocessor
        self.masterIdByBugId = masterIdByBugId

        f = h5py.File(negativeListFile, 'r')

        self.logger.info("Reading HDF5 file {}".format(negativeListFile))
        self.bugIdList = list(f['report_ids'])
        self.idxByReportId = dict([(bug_id, idx) for idx, bug_id in enumerate(self.bugIdList)])
        self.recommendation = f['recommendation']

        self.maxRecommendation = self.recommendation.shape[
            -1] if maxRecommendation is None or maxRecommendation < 1 else maxRecommendation

        self.logger.info("Recommendation list size: {}".format(self.maxRecommendation))
        self.randomAnchor = randomAnchor

    def sampleNegativeExample(self, anchorId):
        anchorIdx = self.idxByReportId[anchorId]
        negList = self.recommendation[anchorIdx]
        idx = random.randint(0, self.maxRecommendation - 1)
        return str(negList[idx])

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        negativePairs = []

        self.setPossibleAnchors(pairsWithId)

        for _ in range(self.rate):
            for idx in range(len(posPairs)):
                anchorId = self.possibleAnchors[random.randint(0, len(self.possibleAnchors) - 1)]

                negativePairs.append(
                    (self.preprocessor.extract(str(anchorId)),
                     self.preprocessor.extract(self.sampleNegativeExample(anchorId)),
                     0))

        self.logger.info(
            "It was generated {} negative pairs. Positive pairs={}".format(len(negativePairs), len(posPairs)))

        return posPairs, negativePairs

    def generateTriplets(self, model, tripletLoss, posPairs):
        """
        Given a positive pair, the class generates the new examples.
        :param model:
        :param loss:
        :param posPairs:
        :return:
        """
        triplets = []

        for anchor in posPairs:
            anchorId, anchorEmb = anchor[0]
            posId, posEmb = anchor[1]

            for _ in range(self.rate):
                negId = self.sampleNegativeExample(int(anchorId))

                triplets.append(anchorEmb, posEmb, self.preprocessor.extract(negId))

        self.logger.info("It was generated {} triplets".format(len(triplets)))

        return triplets


class PositivePreSelectedGenerator(BasicGenerator):

    def __init__(self, negativeListFile, preprocessor, collate, rate, masterIdByBugId, maxRecommendation=None,
                 randomAnchor=True):
        super(PositivePreSelectedGenerator, self).__init__(None, randomAnchor)
        self.logger = logging.getLogger(__name__)
        self.rate = rate
        self.logger = logging.getLogger(__name__)
        self.preprocessor = preprocessor
        self.masterIdByBugId = masterIdByBugId
        self.collate = collate

        f = h5py.File(negativeListFile, 'r')

        self.logger.info("Reading HDF5 file {}".format(negativeListFile))
        self.bugIdList = list(f['report_ids'])
        self.idxByReportId = dict([(bug_id, idx) for idx, bug_id in enumerate(self.bugIdList)])
        self.recommendation = f['recommendation']

        self.maxRecommendation = self.recommendation.shape[
            -1] if maxRecommendation is None or maxRecommendation < 1 else maxRecommendation

        self.logger.info("Recommendation list size: {}".format(self.maxRecommendation))
        # f.close()

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        nNegativePairs = len(posPairs) * self.rate
        negativePairs = []

        self.setPossibleAnchors(pairsWithId)

        self.logger.info("Positive pre selected")

        nBugPerPos = ceil(nNegativePairs / len(self.possibleAnchors))
        random.shuffle(self.possibleAnchors)

        for anchorId in self.possibleAnchors:
            for i, newNegativePair in enumerate(self.generateExamples(model, loss, str(anchorId))):
                negativePairs.append(newNegativePair)

                if i + 1 >= nBugPerPos or len(negativePairs) == nNegativePairs:
                    break

            if len(negativePairs) == nNegativePairs:
                break

        if len(negativePairs) > nNegativePairs:
            msg = "Negative pairs ({}) was generated in quantity higher than the expected ({})".format(
                len(negativePairs),
                nNegativePairs)
            logging.error(msg)
            raise Exception(msg)
        elif len(negativePairs) < len(posPairs) * self.rate:
            self.logger.info(
                "We generated a number of negative pairs (%d) that was insufficient to maintain the same rate." % (
                    len(negativePairs)))
            nPosPairs = int(len(negativePairs) / float(self.rate))
            self.logger.info("Randomly select %d positive pairs to maintain the rate." % (nPosPairs))
            posPairs = random.sample(posPairs, nPosPairs)

        return posPairs, negativePairs

    def generateTriplets(self, model, tripletLoss, posPairs):
        """
        Given a positive pair, the class generates the new examples.
        :param model:
        :param loss:
        :param posPairs:
        :return:
        """
        triplets = []
        nTriplets = len(posPairs) * self.rate

        nBugPerPos = ceil(nTriplets / len(self.possibleAnchors))
        pairIdxs = [i for i in range(len(posPairs))]

        random.shuffle(pairIdxs)

        for idx in pairIdxs:
            for i, newNegativePair in enumerate(self.generateExamples(model, tripletLoss, posPairs[idx])):
                triplets.append(newNegativePair)

                if i + 1 >= nBugPerPos or len(triplets) == nTriplets:
                    break

            if len(triplets) == nTriplets:
                break

        if len(triplets) > nTriplets:
            msg = "Negative pairs ({}) was generated in quantity higher than the expected ({})".format(len(triplets),
                                                                                                       nTriplets)
            logging.error(msg)
            raise Exception(msg)
        elif len(triplets) < len(posPairs) * self.rate:
            self.logger.info(
                "We generated a number of new triplets (%d) that was insufficient to maintain the same rate." % (
                    len(triplets)))

        return triplets

    def generateExamples(self, model, lossFun, anchor):
        batchSize = 128

        if isinstance(anchor, (list, tuple)):
            anchorId, anchorEmb = anchor[0]
            posId, posEmb = anchor[1]
        else:
            anchorId = anchor
            anchorEmb = self.preprocessor.extract(anchorId)
            posId = posEmb = None

        anchorIdx = self.idxByReportId[int(anchorId)]
        negList = self.recommendation[anchorIdx]
        nIteration = math.ceil(len(negList) / float(batchSize))

        model.eval()

        with torch.no_grad():
            for it in range(nIteration):
                batch = []
                bugIds = []

                for negativeReport in negList[it * batchSize: (it + 1) * batchSize]:
                    negId = str(negativeReport)
                    negEmb = self.preprocessor.extract(negId)

                    if posId is None:
                        bugIds.append((anchorId, negId))
                        batch.append((anchorEmb, negEmb, 0.0))
                    else:
                        bugIds.append((anchorId, posId, negId))
                        batch.append((anchorEmb, posEmb, negEmb))

                input, target = self.collate(batch)
                output = model(*input)

                lossValues = lossFun(output, target)
                # lossValues, idxs = torch.sort(lossValue, descending=True)

                lossValues = lossValues.data.cpu().numpy()
                # idxs = idxs.data.cpu().numpy()

                for idx, lossValue in enumerate(lossValues):
                    if np.around(lossValue, decimals=3) > 0.0:
                        yield batch[idx]


class MiscOfflineGenerator(object):

    def __init__(self, generators, rates=None):
        if rates is None:
            prop = 1.0 / len(generators)
            rates = [prop for _ in range(len(generators))]

        start = 0.0

        for i, r in enumerate(rates):
            rates[i] = start + r
            start += r

        if not (0.9999 < start < 1.0001):
            raise Exception("Sum of rates is not 1")

        self.rates = rates
        self.generators = generators
        self.logger = logging.getLogger()

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        pairs_generators = [[] for _ in range(len(self.generators))]

        for posPair in posPairs:
            d = random.random()

            for idx, r in enumerate(self.rates):
                if d < r:
                    pairs_generators[idx].append(posPair)
                    break

        negativePairs = []
        newPositivePairs = []

        for pairs, generator in zip(pairs_generators, self.generators):
            pos, neg = generator.generatePairs(model, loss, pairs, pairsWithId)

            newPositivePairs.extend(pos)
            negativePairs.extend(neg)

            self.logger.info(
                "It was generated {} negative pairs using {}. Positive pairs={}".format(len(neg),
                                                                                        type(generator).__name__,
                                                                                        len(pos)))

        return posPairs, negativePairs
