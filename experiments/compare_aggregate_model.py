"""
This script implements a siamese neural network which extract the information from each bug report of a pair.
The extracted features from each pair is used to calculate the similarity of them or probability of being duplicate.
"""
import logging
import math
import os
from argparse import ArgumentError
from datetime import datetime

import ignite
import torch
import torch.nn.functional as F
from ignite.engine import Events, Engine
from nltk import WhitespaceTokenizer
from pytorch_transformers import BertTokenizer
from sacred import Experiment
from torch import optim
from torch.nn import NLLLoss
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR
from torch.utils.data import DataLoader

from classical_approach.bm25f import SUN_REPORT_ID_INDEX
from classical_approach.read_data import read_weights, read_dbrd_file
from data.Embedding import Embedding
from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from data.collate import PairBugCollate, LazyPairCollate
from data.dataset import PairBugDatasetReader, LazyPairBugDatasetReader
from data.input_handler import RNNInputHandler, BERTInputHandler
from data.preprocessing import PreprocessingCache, \
    SummaryPreprocessor, MultiLineTokenizer, \
    loadFilters, DescriptionPreprocessor, ElmoPreprocessorList, TransformerPreprocessor, Preprocessor, PreprocessorList
from example_generator.offline_pair_geneneration import NonNegativeRandomGenerator, RandomGenerator, KRandomGenerator, \
    MiscNonZeroRandomGen, PreSelectedGenerator, MiscOfflineGenerator, PositivePreSelectedGenerator, REPGenerator
from metrics.metric import AverageLoss, MeanScoreDistance
from metrics.ranking import PreselectListRanking, DeshmukhRanking, GeneralScorer, CompareAggregateScorer, SunRanking, \
    REP_CADD_Recommender, generateRecommendationList
from model.compare_aggregate import CADD
from util.jsontools import JsonLogFormatter
from util.siamese_util import processCategoricalParam
from util.training_loop_util import logMetrics, logRankingResult

ex = Experiment("filter_model")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logHandler = logging.StreamHandler()
formatter = JsonLogFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

ex.logger = logger


@ex.config
def cfg():
    # Set here all possible parameters; You have to change some parameters values.
    bug_database = None
    epochs = 20
    lr = 0.001
    l2 = 0.0
    batch_size = 16
    cuda = True
    cache_folder = None
    pairs_training = None
    pairs_validation = None
    neg_pair_generator = {
        "type": "none",
        "training": None,
        "rate": 1,
        "pre_list_file": None,
        "k": 0,
        "n_tries": 0,
        "preselected_length": None,
        "random_anchor": True
    }

    compare_aggregation = {
        "word_embedding": None,
        "sent_representation": None,  # Sentence representation (each word of the sentence is represented by vector)
        "tokenizer": None,
        "only_candidate": False,
        "summary": {
            'update_embedding': False,
            "filters": ["TransformLowerCaseFilter"],
            'model_type': 'lstm',
            'hidden_size': 100,
            'bidirectional': False,
            'num_layers': 1,
            'dropout': 0.0,
            "fine_tune": False,
            "layer_norm": False,
            'residual': False
        },
        "desc": {
            'summarization': False,
            'update_embedding': False,
            "filters": ["TransformLowerCaseFilter"],
            'model_type': 'lstm',
            'hidden_size': 100,
            'bidirectional': False,
            'num_layers': 1,
            'dropout': 0.0,
            "fine_tune": False,
            "layer_norm": False,
            'residual': False
        },

        "batch_normalization": False,
        "matching": {
            "cross_attention": True,
            "attention": "general",
            "comparison_function": "sub_mult_nn",
            "comparison_hidden_size": 100,
            "categorical_hidden_layer": 10,
            "categorical_layer_norm": False,
            'dropout': 0.0,
            'categorical_dropout': 0.0,
            'residual': False,
            "layer_norm": False,
        },
        "aggregate": {
            "model": "cnn",
            "window": [3, 4, 5],
            "nfilters": 100,
            "hidden_size": 100,
            'dropout': 0.0,
            'bidirectional': True,
            'num_layers': 1,
            'model_type': 'lstm',
            'pooling': 'self_att',
            'self_att_hidden': 50,
            'layer_norm': False,
            'residual': False,
            'concat': False
        },
        "hidden_size": 300,
        'dropout': 0.0,
        'layer_norm': False,
        'cat_layer_norm': False
    }

    categorical = {
        "lexicons": None,
        "bn_last_layer": False,
        "emb_size": 20,
        "hidden_sizes": None,
        "dropout": 0.0,
        "activation": None,
        "batch_normalization": False,
        "layer_norm": False
    }
    random_switch = False
    recall_estimation_train = None
    recall_estimation = None
    sample_size_rr_tr = 0  # Number of bug report in the training that will be sample to evaluate the recall rate
    sample_size_rr_val = 0  # Number of bug report in the validation that will be sample to evaluate the recall rate
    rr_val_epoch = 1
    rr_train_epoch = 5
    ranking_result_file = None
    optimizer = "adam"
    lr_scheduler = {
        "type": "linear",
        "decay": 1,
        "step_size": 1
    }
    save = None
    save_by_epoch = None
    load = None
    recall_rate = {
        'type': 'none',  # 3 options: none, sun2011 and deshmukh
        'dataset': None,
        'result_file': None,
        'group_by_master': True,
        'window': None  # only compare bug that are in this interval of days

        # File where we store the position of each duplicate bug in the list, the first 30 top reports,
    }
    rep ={
        'model': None,
        'input': None,
        'training': None,
        'rate': 1,
        'neg_training': 10000,
        'k': 1000
    }
    rr_scorer = "general"  # Options: general or optimized


@ex.automain
def main(_run, _config, _seed, _log):
    """

    :param _run:
    :param _config:
    :param _seed:
    :param _log:
    :return:
    """

    """
    Setting and loading parameters
    """
    # Setting logger
    args = _config
    logger = _log

    logger.info(args)
    logger.info('It started at: %s' % datetime.now())

    torch.manual_seed(_seed)

    bugReportDatabase = BugReportDatabase.fromJson(args['bug_database'])
    paddingSym = "</s>"
    batchSize = args['batch_size']

    device = torch.device('cuda' if args['cuda'] else "cpu")

    if args['cuda']:
        logger.info("Turning CUDA on")
    else:
        logger.info("Turning CUDA off")

    # It is the folder where the preprocessed information will be stored.
    cacheFolder = args['cache_folder']

    # Setting the parameter to save and loading parameters
    importantParameters = ['compare_aggregation', 'categorical']
    parametersToSave = dict([(parName, args[parName]) for parName in importantParameters])

    if args['load'] is not None:
        mapLocation = (lambda storage, loc: storage.cuda()) if cudaOn else 'cpu'
        modelInfo = torch.load(args['load'], map_location=mapLocation)
        modelState = modelInfo['model']

        for paramName, paramValue in modelInfo['params'].items():
            args[paramName] = paramValue
    else:
        modelState = None

    if args['rep'] is not None and args['rep']['model']:
        logger.info("Loading REP")
        rep = read_weights(args['rep']['model'])
        rep_input, max_tkn_id = read_dbrd_file(args['rep']['input'], math.inf)
        rep_recommendation = args['rep']['k']

        rep.fit_transform(rep_input,max_tkn_id, True)

        rep_input_by_id = {}

        for inp in rep_input:
            rep_input_by_id[inp[SUN_REPORT_ID_INDEX]] = inp

    else:
        rep = None

    preprocessors = PreprocessorList()
    inputHandlers = []

    categoricalOpt = args.get('categorical')

    if categoricalOpt is not None and len(categoricalOpt) != 0:
        categoricalEncoder, _, _ = processCategoricalParam(categoricalOpt, bugReportDatabase, inputHandlers,
                                                           preprocessors, None, logger,
                                                           cudaOn)
    else:
        categoricalEncoder = None

    filterInputHandlers = []

    compareAggOpt = args['compare_aggregation']
    databasePath = args['bug_database']

    # Loading word embedding
    if compareAggOpt["word_embedding"]:
        # todo: Allow use embeddings and other representation
        lexicon, embedding = Embedding.fromFile(compareAggOpt['word_embedding'], 'UUUKNNN', hasHeader=False,
                                                paddingSym=paddingSym)
        logger.info("Lexicon size: %d" % (lexicon.getLen()))
        logger.info("Word Embedding size: %d" % (embedding.getEmbeddingSize()))
        paddingId = lexicon.getLexiconIndex(paddingSym)
        lazy = False
    else:
        embedding = None

    # Tokenizer
    if compareAggOpt['tokenizer'] == 'default':
        logger.info("Use default tokenizer to tokenize summary information")
        tokenizer = MultiLineTokenizer()
    elif compareAggOpt['tokenizer'] == 'white_space':
        logger.info("Use white space tokenizer to tokenize summary information")
        tokenizer = WhitespaceTokenizer()
    else:
        raise ArgumentError(
            "Tokenizer value %s is invalid. You should choose one of these: default and white_space" %
            compareAggOpt['tokenizer'])

    # Preparing input handlers, preprocessors and cache
    minSeqSize = max(compareAggOpt['aggregate']["window"]) if compareAggOpt['aggregate']["model"] == "cnn" else -1

    if compareAggOpt['summary'] is not None:
        # Use summary and description (concatenated) to address this problem
        logger.info("Using Summary information.")
        # Loading Filters
        sumFilters = loadFilters(compareAggOpt['summary']['filters'])

        if compareAggOpt['summary']['model_type'] in ('lstm', 'gru', 'word_emd', 'residual'):
            arguments = (
                databasePath, compareAggOpt['word_embedding'],
                ' '.join(sorted([fil.__class__.__name__ for fil in sumFilters])),
                compareAggOpt['tokenizer'], SummaryPreprocessor.__name__)

            inputHandlers.append(RNNInputHandler(paddingId, minInputSize=minSeqSize))

            summaryCache = PreprocessingCache(cacheFolder, arguments)
            summaryPreprocessor = SummaryPreprocessor(lexicon, bugReportDatabase, sumFilters, tokenizer, paddingId,
                                                      summaryCache)
        elif compareAggOpt['summary']['model_type'] == 'ELMo':
            raise NotImplementedError("ELMO is not implemented!")
            # inputHandlers.append(ELMoInputHandler(cudaOn, minInputSize=minSeqSize))
            # summaryPreprocessor = ELMoPreprocessor(0, elmoEmbedding)
            # compareAggOpt['summary']["input_size"] = elmoEmbedding.get_size()
        elif compareAggOpt['summary']['model_type'] == 'BERT':
            arguments = (databasePath, "CADD SUMMARY", "BERT", "bert-base-uncased")

            inputHandlers.append(BERTInputHandler(0, minInputSize=minSeqSize))

            summaryCache = PreprocessingCache(cacheFolder, arguments)
            summaryPreprocessor = TransformerPreprocessor("short_desc", "bert-base-uncased", BertTokenizer, 0,
                                                          bugReportDatabase, summaryCache)
#            compareAggOpt['summary']["input_size"] = 768

        preprocessors.append(summaryPreprocessor)

    if compareAggOpt['desc'] is not None:
        # Use summary and description (concatenated) to address this problem
        logger.info("Using Description information.")
        descFilters = loadFilters(compareAggOpt['desc']['filters'])

        if compareAggOpt['desc']['model_type'] in ('lstm', 'gru', 'word_emd', 'residual'):
            arguments = (
                databasePath, compareAggOpt['word_embedding'],
                ' '.join(sorted([fil.__class__.__name__ for fil in descFilters])),
                compareAggOpt['tokenizer'], "CADD DESC", str(compareAggOpt['desc']['summarization']))

            inputHandlers.append(RNNInputHandler(paddingId, minInputSize=minSeqSize))

            descriptionCache = PreprocessingCache(cacheFolder, arguments)
            descPreprocessor = DescriptionPreprocessor(lexicon, bugReportDatabase, descFilters, tokenizer, paddingId,
                                                       cache=descriptionCache)
        elif compareAggOpt['desc']['model_type'] == 'ELMo':
            raise NotImplementedError("ELMO is not implemented!")
            # inputHandlers.append(ELMoInputHandler(cudaOn, minInputSize=minSeqSize))
            # descPreprocessor = ELMoPreprocessor(1, elmoEmbedding)
            # compareAggOpt['desc']["input_size"] = elmoEmbedding.get_size()
        elif compareAggOpt['desc']['model_type'] == 'BERT':
            arguments = (databasePath, "CADD DESC", "BERT", "bert-base-uncased")

            inputHandlers.append(BERTInputHandler(0, minInputSize=minSeqSize))

            descriptionCache = PreprocessingCache(cacheFolder, arguments)
            descPreprocessor = TransformerPreprocessor("description", "bert-base-uncased", BertTokenizer, 0,
                                                       bugReportDatabase, descriptionCache)
#            compareAggOpt['desc']["input_size"] = 768

        preprocessors.append(descPreprocessor)

    # Create model
    model = CADD(embedding, categoricalEncoder, compareAggOpt, compareAggOpt['summary'], compareAggOpt['desc'],
                 compareAggOpt['matching'], compareAggOpt['aggregate'], cudaOn=cudaOn)

    lossFn = F.nll_loss
    lossNoReduction = NLLLoss(reduction='none')

    if cudaOn:
        model.cuda()

    if modelState:
        model.load_state_dict(modelState)

    """
    Loading the training and validation. Also, it sets how the negative example will be generated.
    """
    cmpAggCollate = PairBugCollate(inputHandlers, torch.int64)

    # load training
    if args.get('pairs_training'):
        negativePairGenOpt = args.get('neg_pair_generator', )
        pairTrainingFile = args.get('pairs_training')

        offlineGeneration = not (negativePairGenOpt is None or negativePairGenOpt['type'] == 'none')
        masterIdByBugId = bugReportDatabase.getMasterIdByBugId()
        randomAnchor = negativePairGenOpt['random_anchor']

        if rep:
            logger.info("Generate negative examples using REP.")
            randomAnchor = negativePairGenOpt['random_anchor']
            trainingDataset = BugDataset(args['rep']['training'])

            bugIds = trainingDataset.bugIds
            negativePairGenerator = REPGenerator(rep, rep_input_by_id, args['rep']['neg_training'], preprocessors,
                                                 bugIds, masterIdByBugId, args['rep']['rate'], randomAnchor)
        elif not offlineGeneration:
            logger.info("Not generate dynamically the negative examples.")
            negativePairGenerator=None
        else:
            pairGenType = negativePairGenOpt['type']

            if pairGenType == 'random':
                logger.info("Random Negative Pair Generator")
                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        trainingDataset.info, len(bugIds)))

                negativePairGenerator = RandomGenerator(preprocessors, cmpAggCollate,
                                                        negativePairGenOpt['rate'],
                                                        bugIds, masterIdByBugId, randomAnchor=randomAnchor)

            elif pairGenType == 'non_negative':
                logger.info("Non Negative Pair Generator")
                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        trainingDataset.info, len(bugIds)))

                negativePairGenerator = NonNegativeRandomGenerator(preprocessors, cmpAggCollate,
                                                                   negativePairGenOpt['rate'],
                                                                   bugIds, masterIdByBugId,
                                                                   negativePairGenOpt['n_tries'], device,
                                                                   randomAnchor=randomAnchor)
            elif pairGenType == 'misc_non_zero':
                logger.info("Misc Non Zero Pair Generator")
                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        trainingDataset.info, len(bugIds)))

                negativePairGenerator = MiscNonZeroRandomGen(preprocessors, cmpAggCollate,
                                                             negativePairGenOpt['rate'], bugIds,
                                                             trainingDataset.duplicateIds, masterIdByBugId,
                                                             negativePairGenOpt['n_tries'], device, randomAnchor=randomAnchor)
            elif pairGenType == 'random_k':
                logger.info("Random K Negative Pair Generator")
                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        trainingDataset.info, len(bugIds)))

                negativePairGenerator = KRandomGenerator(preprocessors, cmpAggCollate,
                                                         negativePairGenOpt['rate'],
                                                         bugIds, masterIdByBugId, negativePairGenOpt['k'], device,
                                                         randomAnchor=randomAnchor)
            elif pairGenType == "pre":
                logger.info("Pre-selected list generator")
                negativePairGenerator = PreSelectedGenerator(negativePairGenOpt['pre_list_file'], preprocessors,
                                                             negativePairGenOpt['rate'], masterIdByBugId,
                                                             negativePairGenOpt['preselected_length'],
                                                             randomAnchor=randomAnchor)

            elif pairGenType == "positive_pre":
                logger.info("Positive Pre-selected list generator")
                negativePairGenerator = PositivePreSelectedGenerator(negativePairGenOpt['pre_list_file'],
                                                                     preprocessors, cmpAggCollate,
                                                                     negativePairGenOpt['rate'], masterIdByBugId,
                                                                     negativePairGenOpt['preselected_length'],
                                                                     randomAnchor=randomAnchor)
            elif pairGenType == "misc_non_zero_pre":
                logger.info("Misc: non-zero and Pre-selected list generator")
                negativePairGenerator1 = PreSelectedGenerator(negativePairGenOpt['pre_list_file'], preprocessors,
                                                              negativePairGenOpt['rate'], masterIdByBugId,
                                                              negativePairGenOpt['preselected_length'],
                                                              randomAnchor=randomAnchor)

                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                negativePairGenerator2 = NonNegativeRandomGenerator(preprocessors, cmpAggCollate,
                                                                    negativePairGenOpt['rate'],
                                                                    bugIds, masterIdByBugId,
                                                                    negativePairGenOpt['n_tries'], device,
                                                                    randomAnchor=randomAnchor)

                negativePairGenerator = MiscOfflineGenerator((negativePairGenerator1, negativePairGenerator2))
            elif pairGenType == "misc_non_zero_positive_pre":
                logger.info("Misc: non-zero and Positive Pre-selected list generator")
                negativePairGenerator1 = PositivePreSelectedGenerator(negativePairGenOpt['pre_list_file'],
                                                                      preprocessors, cmpAggCollate,
                                                                      negativePairGenOpt['rate'], masterIdByBugId,
                                                                      negativePairGenOpt['preselected_length'],
                                                                      randomAnchor=randomAnchor)

                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                negativePairGenerator2 = NonNegativeRandomGenerator(preprocessors, cmpAggCollate,
                                                                    negativePairGenOpt['rate'],
                                                                    bugIds, masterIdByBugId,
                                                                    negativePairGenOpt['n_tries'], device,
                                                                    randomAnchor=randomAnchor)

                negativePairGenerator = MiscOfflineGenerator((negativePairGenerator1, negativePairGenerator2))

            else:
                raise ArgumentError(
                    "Offline generator is invalid (%s). You should choose one of these: random, hard and pre" %
                    pairGenType)

        pairTrainingReader = PairBugDatasetReader(pairTrainingFile, preprocessors, negativePairGenerator,
                                                  randomInvertPair=args['random_switch'])
        trainingCollate = cmpAggCollate
        trainingLoader = DataLoader(pairTrainingReader, batch_size=batchSize, collate_fn=trainingCollate.collate,
                                    shuffle=True)
        logger.info("Training size: %s" % (len(trainingLoader.dataset)))

    # load validation
    if args.get('pairs_validation'):
        pairValidationReader = PairBugDatasetReader(args.get('pairs_validation'), preprocessors)
        validationLoader = DataLoader(pairValidationReader, batch_size=batchSize, collate_fn=cmpAggCollate.collate)

        logger.info("Validation size: %s" % (len(validationLoader.dataset)))
    else:
        validationLoader = None

    """
    Training and evaluate the model. 
    """
    optimizer_opt = args.get('optimizer', 'adam')

    if optimizer_opt == 'sgd':
        logger.info('SGD')
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['l2'])
    elif optimizer_opt == 'adam':
        logger.info('Adam')
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2'])

    # Recall rate
    rankingScorer = GeneralScorer(model, preprocessors, device, cmpAggCollate)
    recallEstimationTrainOpt = args.get('recall_estimation_train')

    if recallEstimationTrainOpt:
        preselectListRankingTrain = PreselectListRanking(recallEstimationTrainOpt, args['sample_size_rr_tr'])

    recallEstimationOpt = args.get('recall_estimation')

    if recallEstimationOpt:
        preselectListRanking = PreselectListRanking(recallEstimationOpt, args['sample_size_rr_val'])

    # LR scheduler
    lrSchedulerOpt = args.get('lr_scheduler', None)

    if lrSchedulerOpt is None:
        logger.info("Scheduler: Constant")
        lrSched = None
    elif lrSchedulerOpt["type"] == 'step':
        logger.info("Scheduler: StepLR (step:%s, decay:%f)" % (lrSchedulerOpt["step_size"], args["decay"]))
        lrSched = StepLR(optimizer, lrSchedulerOpt["step_size"], lrSchedulerOpt["decay"])
    elif lrSchedulerOpt["type"] == 'exp':
        logger.info("Scheduler: ExponentialLR (decay:%f)" % (lrSchedulerOpt["decay"]))
        lrSched = ExponentialLR(optimizer, lrSchedulerOpt["decay"])
    elif lrSchedulerOpt["type"] == 'linear':
        logger.info("Scheduler: Divide by (1 + epoch * decay) ---- (decay:%f)" % (lrSchedulerOpt["decay"]))

        lrDecay = lrSchedulerOpt["decay"]
        lrSched = LambdaLR(optimizer, lambda epoch: 1 / (1.0 + epoch * lrDecay))
    else:
        raise ArgumentError(
            "LR Scheduler is invalid (%s). You should choose one of these: step, exp and linear " %
            pairGenType)

    # Set training functions
    def trainingIteration(engine, batch):
        engine.kk = 0

        model.train()
        optimizer.zero_grad()
        x, y = batch
        output = model(*x)
        loss = lossFn(output, y)
        loss.backward()
        optimizer.step()
        return loss, output, y

    def scoreDistanceTrans(output):
        if len(output) == 3:
            _, y_pred, y = output
        else:
            y_pred, y = output

        if lossFn == F.nll_loss:
            return torch.exp(y_pred[:, 1]), y

    trainer = Engine(trainingIteration)
    trainingMetrics = {'training_loss': AverageLoss(lossFn, batch_size=lambda x: x[0].shape[0]),
                       'training_dist_target': MeanScoreDistance(output_transform=scoreDistanceTrans)}

    # Add metrics to trainer
    for name, metric in trainingMetrics.items():
        metric.attach(trainer, name)

    # Set validation functions
    def validationIteration(engine, batch):
        if not hasattr(engine, 'kk'):
            engine.kk = 0

        model.eval()
        with torch.no_grad():
            x, y = batch
            y_pred = model(*x)

            # for k, (pred, t) in enumerate(zip(y_pred, y)):
            #     engine.kk += 1
            #     print("{}: {} \t {}".format(engine.kk, torch.round(torch.exp(pred) * 100), t))
            return y_pred, y

    validationMetrics = {'validation_loss': ignite.metrics.Loss(lossFn),
                         'validation_dist_target': MeanScoreDistance(output_transform=scoreDistanceTrans)}
    evaluator = Engine(validationIteration)

    # Add metrics to evaluator
    for name, metric in validationMetrics.items():
        metric.attach(evaluator, name)

    # recommendation
    if rep:
        recommendation_fn = REP_CADD_Recommender(rep, rep_input_by_id, rep_recommendation).generateRecommendationList
    else:
        recommendation_fn = generateRecommendationList

    @trainer.on(Events.EPOCH_STARTED)
    def onStartEpoch(engine):
        epoch = engine.state.epoch
        logger.info("Epoch: %d" % epoch)

        if lrSched:
            lrSched.step()

        logger.info("LR: %s" % str(optimizer.param_groups[0]["lr"]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def onEndEpoch(engine):
        epoch = engine.state.epoch

        logMetrics(_run, logger, engine.state.metrics, epoch)

        # Evaluate Training
        if validationLoader:
            evaluator.run(validationLoader)
            logMetrics(_run, logger, evaluator.state.metrics, epoch)

        if recallEstimationTrainOpt and (epoch % args['rr_train_epoch'] == 0):
            logRankingResult(_run, logger, preselectListRankingTrain, rankingScorer, bugReportDatabase, None, epoch,
                             "train", recommendationListfn=recommendation_fn)
            rankingScorer.free()

        if recallEstimationOpt and (epoch % args['rr_val_epoch'] == 0):
            logRankingResult(_run, logger, preselectListRanking, rankingScorer, bugReportDatabase,
                             args.get("ranking_result_file"), epoch, "validation", recommendationListfn=recommendation_fn)
            rankingScorer.free()

        pairTrainingReader.sampleNewNegExamples(model, lossNoReduction)

        if args.get('save'):
            save_by_epoch = args['save_by_epoch']

            if save_by_epoch and epoch in save_by_epoch:
                file_name, file_extension = os.path.splitext(args['save'])
                file_path = file_name + '_epoch_{}'.format(epoch) + file_extension
            else:
                file_path = args['save']

            modelInfo = {'model': model.state_dict(),
                         'params': parametersToSave}

            logger.info("==> Saving Model: %s" % file_path)
            torch.save(modelInfo, file_path)

    if args.get('pairs_training'):
        trainer.run(trainingLoader, max_epochs=args['epochs'])
    elif args.get('pairs_validation'):
        # Evaluate Training
        evaluator.run(validationLoader)
        logMetrics(_run, logger, evaluator.state.metrics, 0)

        if recallEstimationOpt:
            logRankingResult(_run, logger, preselectListRanking, rankingScorer, bugReportDatabase,
                             args.get("ranking_result_file"), 0, "validation", recommendationListfn=recommendation_fn)

    recallRateOpt = args.get('recall_rate', {'type': 'none'})
    if recallRateOpt['type'] != 'none':
        if recallRateOpt['type'] == 'sun2011':
            logger.info("Calculating recall rate: {}".format(recallRateOpt['type']))
            recallRateDataset = BugDataset(recallRateOpt['dataset'])

            rankingClass = SunRanking(bugReportDatabase, recallRateDataset, recallRateOpt['window'])
            # We always group all bug reports by master in the results in the sun 2011 methodology
            group_by_master = True
        elif recallRateOpt['type'] == 'deshmukh':
            logger.info("Calculating recall rate: {}".format(recallRateOpt['type']))
            recallRateDataset = BugDataset(recallRateOpt['dataset'])
            rankingClass = DeshmukhRanking(bugReportDatabase, recallRateDataset)
            group_by_master = recallRateOpt['group_by_master']
        else:
            raise ArgumentError(
                "recall_rate.type is invalid (%s). You should choose one of these: step, exp and linear " %
                recallRateOpt['type'])

        logRankingResult(_run, logger, rankingClass, rankingScorer, bugReportDatabase,
                         recallRateOpt["result_file"], 0, None, group_by_master, recommendationListfn=recommendation_fn)
