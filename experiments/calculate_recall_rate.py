import argparse
import logging
import pickle
from time import time

from data.bug_report_database import BugReportDatabase
from data.bug_dataset import BugDataset
from metrics.ranking import TFIDFScorer, SharedEncoderNNScorer, generateRecomendationList, generateRecommendationListParalell, \
    RecallRateEstimator
from data.preprocessing import ClassicalPreprocessing
from util.jsontools import JsonLogFormatter


class Obj(object):

    def __init__(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recall_ratio_k', nargs='+', required=True,
                        help="list of the values of k to be used in the recall ratio. If k is empty list so recall rate "
                             "is not calculated")
    parser.add_argument('--bug_dataset', help="")
    parser.add_argument('--input', required=True)

    parser.add_argument('--model', help="model")
    parser.add_argument('--model_type', help="model type")
    parser.add_argument('--cuda', action="store_true", help="enable cuda.")
    parser.add_argument('--nb_processes', type=int, default=8, help="")
    parser.add_argument('--only_master', action="store_true",
                        help="Only compare the new bugs with the master sets.")
    parser.add_argument('--sorted_by_date', action="store_true",
                        help="")
    parser.add_argument('--rm_special_chars', action="store_true", help="Remove non-alpha-numeric characters.")
    parser.add_argument('--recall_estimation',
                        help="Performs the recall rate estimation each epoch. This parameter receives the file that contains the list of bug ids.")


    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logHandler = logging.StreamHandler()
    formatter = JsonLogFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    args = parser.parse_args()

    logger.info(args.__dict__)

    args.recall_ratio_k = [int(k) for k in args.recall_ratio_k]
    bugSetDataset = BugDataset(args.input)
    bugReportDatabase = BugReportDatabase.fromJson(args.bug_dataset)

    if args.recall_estimation:
        bugIds, listByBugId = pickle.load(open(args.recall_estimation, 'rb'))
        duplicateBugs = list(listByBugId.keys())
    else:
        listByBugId= None
        bugIds = []

        for idx in range(len(bugReportDatabase)):
            bugIds.append(bugReportDatabase.getBugByIndex(idx)['bug_id'])

        duplicateBugs = bugSetDataset.duplicateIdxs


    similarityListByDuplicate = []

    if args.model_type == 'tfidf':
        # We have to import those module to load the model
        from nltk import TreebankWordTokenizer, SnowballStemmer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from nltk.corpus import stopwords
        from data.preprocessing import ClassicalPreprocessing, MultiLineTokenizer, \
            StripPunctuactionFilter, DectectNotUsualWordFilter, TransformNumberToZeroFilter

        recallAux = TFIDFScorer(pickle.load(open(args.model, 'rb')))
    else:
        # We can't import torch without allocating a GPU in Cedar cluster.
        from experiments.duplicate_bug_detection_deep_learning import getDataHandlerLexiconEmb, createNeuralNetworkModel
        from data.dataset import BugDataExtractor

        # Load Model and DataHandlers
        arguments = Obj({
            'load': args.model,
            'cuda': args.cuda,
            'rm_special_chars': args.rm_special_chars,
            'description_bn': True,
            'classifier_bn': True,
            'classifier_only_mult_dif': False,
        })

        dataHandlers, lexicons, embeddings, arguments = getDataHandlerLexiconEmb(arguments)
        encoderContainer, model = createNeuralNetworkModel(dataHandlers, lexicons, embeddings, arguments)

        dataExtractor = BugDataExtractor(bugReportDatabase, dataHandlers)
        recallAux = SharedEncoderNNScorer(dataExtractor, arguments.model, encoderContainer, model, args.cuda)

    # Send message to pre-generate bug embedding
    begin = time()
    logger.info("Generating features vectors")
    scorer.pregenerateBugEmbedding(bugIds)
    bugEmbeddingById = scorer.pregenerateBugEmbedding(bugReportDatabase, bugIds)
    logger.info("Time spent to generate features vectors: %f" % (time() - begin))


    masterSetById = bugReportDatabase.getMasterSetById()

    recallKs = sorted(args.recall_ratio_k)
    if args.model_type == 'tfidf':
        begin = time()
        logger.info("Generating features vectors")

        bugEmbeddingById = recallAux.pregenerateBugEmbedding(bugReportDatabase, bugIds)
        logger.info("Time spent to generate features vectors: %f" % (time() - begin))

        hitsPerRateK, total = generateRecommendationListParalell(args.nb_processes, recallKs, recallAux, bugIds, bugSetDataset.duplicateIds,
                                                                 bugReportDatabase, masterSetById, bugEmbeddingById, onlyMaster=args.only_master)
    else:
        hitsPerRateK, total = generateRecomendationList(recallKs, recallAux, bugIds, bugSetDataset.duplicateIds,
                                                        bugReportDatabase, masterSetById, onlyMaster=args.only_master)

    logger.info("Recall Rate Results:")
    for k, hit in zip(recallKs, hitsPerRateK):
        rate = float(hit) / total
        logger.info({'type': "metric", 'label': 'recall_rate', 'k': k, 'rate': rate, 'hit': hit, 'total': total})