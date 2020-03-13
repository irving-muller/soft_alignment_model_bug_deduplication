"""
This script implements a siamese neural network which extract the information from each bug report of a pair.
The extracted features from each pair is used to calculate the similarity of them or probability of being duplicate.
"""
import logging
from argparse import ArgumentError
from datetime import datetime

import ignite
import torch
from ignite.engine import Events, Engine
from sacred import Experiment
from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR
from torch.utils.data import DataLoader

from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from data.collate import TripletBugCollate
from data.dataset import TripletBugDatasetReader
from data.preprocessing import PreprocessorList
from example_generator.offline_pair_geneneration import NonNegativeRandomGenerator, PreSelectedGenerator, \
    RandomGenerator, KRandomGenerator, MiscOfflineGenerator
from metrics.metric import AverageLoss, LossWrapper
from metrics.ranking import PreselectListRanking, SharedEncoderNNScorer, \
    DeshmukhRanking, SunRanking
from model.loss import TripletLoss
from model.siamese import CosineTripletNN
from util.jsontools import JsonLogFormatter
from util.siamese_util import processSumDescParam, processSumParam, processCategoricalParam, processDescriptionParam
from util.training_loop_util import logMetrics, logRankingResult

ex = Experiment("siamese_triplets")

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
    l2 = 0.00
    batch_size = 16
    ranking_batch_size = 256
    cuda = True
    cache_folder = None
    triplets_training = None
    triplets_validation = None
    neg_pair_generator = {
        "type": "none",
        "training": None,
        "rate": 1,
        "pre_list_file": None,
        "k": 0,
        "n_tries": 0,
        "preselected_length": None,
        "decimals": 3
    }
    sum_desc = {
        "word_embedding": None,
        "lexicon": None,
        "tokenizer": None,
        "filters": ["TransformLowerCaseFilter"],
        "encoder_type": "cnn",
        "window_sizes": [3],
        "nfilters": 100,
        "update_embedding": False,
        "activation": "relu",
        "batch_normalization": False,
        "dropout": 0.0,
        "hidden_sizes": None,
        "hidden_act": None,
        "hidden_dropout": 0.0,
        "bn_last_layer": False
    }
    summary = {
        "word_embedding": None,
        "lexicon": None,
        "encoder_type": None,
        "tokenizer": None,
        "filters": ["TransformLowerCaseFilter"],
        'rnn_type': None,
        'hidden_size': 100,
        'bidirectional': False,
        'num_layers': 1,
        'update_embedding': False,
        'fixed_opt': 'mean',
        "activation": "relu",
        "batch_normalization": False,
        "dropout": 0.0,
        "hidden_sizes": None,
        "hidden_act": None,
        "hidden_dropout": 0.0,
        "bn_last_layer": False,
        "window_sizes": [3],
        "self_att_hidden": 100,
        "n_hops": 20
    }

    description = {
        "word_embedding": None,
        "lexicon": None,
        "encoder_type": None,
        "tokenizer": None,
        "filters": ["TransformLowerCaseFilter"],
        'rnn_type': None,
        'hidden_size': 100,
        'bidirectional': False,
        'num_layers': 1,
        'dropout': 0.0,
        'update_embedding': False,
        'fixed_opt': 'mean',
        "activation": "relu",
        "batch_normalization": False,
        "window_sizes": [3],
        "nfilters": 100,
        "hidden_sizes": None,
        "hidden_act": None,
        "hidden_dropout": 0.0,
        "bn_last_layer": False,
        "self_att_hidden": 100,
        "n_hops": 20
    }

    categorical = {
        "lexicons": None,
        "bn_last_layer": False,
        "emb_size": 20,
        "hidden_sizes": None,
        "dropout": 0.0,
        "activation": None,
        "batch_normalization": False,
    }
    scorer = {
        "type": "binary",
        "without_embedding": False,
        "batch_normalization": False,
        "hidden_sizes": [100, 200],
        "margin": 0,
        "loss": None,
        "dropout": 0
    }
    recall_estimation_train = None
    recall_estimation = None
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
    load = None
    recall_rate = {
        'type': 'none',  # 3 options: none, sun2011 and deshmukh
        'dataset': None,
        'result_file': None,
        'group_by_master': True,
        'window': None  # only compare bug that are in this interval of days

        # File where we store the position of each duplicate bug in the list, the first 30 top reports,
    }


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
    importantParameters = ['summary', 'description', 'sum_desc', 'scorer', 'categorical']
    parametersToSave = dict([(parName, args[parName]) for parName in importantParameters])

    if args['load'] is not None:
        mapLocation = (lambda storage, loc: storage.cuda()) if args['cuda'] else 'cpu'
        modelInfo = torch.load(args['load'], map_location=mapLocation)
        modelState = modelInfo['model']

        for paramName, paramValue in modelInfo['params'].items():
            args[paramName] = paramValue
    else:
        modelState = None

    """
    Set preprocessor that will pre-process the raw information from the bug reports.
    Each different information has a specific encoder(NN), preprocessor and input handler.
    """

    preprocessors = PreprocessorList()
    encoders = []
    inputHandlers = []

    sum_desc_opts = args['sum_desc']
    databasePath = args['bug_database']

    if sum_desc_opts is not None:
        processSumDescParam(sum_desc_opts, bugReportDatabase, inputHandlers, preprocessors, encoders, cacheFolder,
                            databasePath, logger, paddingSym)

    sumOpts = args.get("summary")

    if sumOpts is not None:
        processSumParam(sumOpts, bugReportDatabase, inputHandlers, preprocessors, encoders, databasePath, cacheFolder,
                        logger, paddingSym)

    descOpts = args.get("description")

    if descOpts is not None:
        processDescriptionParam(descOpts, bugReportDatabase, inputHandlers, preprocessors, encoders, databasePath,
                                cacheFolder, logger, paddingSym)

    categoricalOpt = args.get('categorical')
    if categoricalOpt is not None and len(categoricalOpt) != 0:
        processCategoricalParam(categoricalOpt, bugReportDatabase, inputHandlers, preprocessors, encoders, logger)

    """
    Set the final scorer and the loss. Load the scorer if this argument was set.
    """
    scorerOpts = args['scorer']
    scorerType = scorerOpts['type']

    if scorerType == 'binary':
        pass
        # withoutBugEmbedding = scorerOpts.get('without_embedding', False)
        # batchNorm = scorerOpts.get('batch_normalization', True)
        # hiddenSizes = scorerOpts.get('hidden_sizes', [100])
        # model = ProbabilityPairNN(encoders, withoutBugEmbedding, hiddenSizes, batchNorm)
        # lossFn = BCELoss()
        # lossNoReduction = BCELoss(reduction='none')
        #
        # logger.info("Using BCELoss")
    elif scorerType == 'cosine':
        model = CosineTripletNN(encoders, scorerOpts['dropout'])
        margin = scorerOpts.get('margin', 0.0)

        if (categoricalOpt is not None and categoricalOpt.get('bn_last_layer', False)) or (
                sum_desc_opts is not None and sum_desc_opts.get('bn_last_layer', False)) or (
                sumOpts is not None and sumOpts.get('bn_last_layer')):
            raise Exception('You are applying batch normalization in the bug embedding.')

        lossFn = TripletLoss(margin)
        lossNoReduction = TripletLoss(margin, reduction='none')
        logger.info("Using Cosine Embeding Loss: margin={}".format(margin))

    model.to(device)

    if modelState:
        model.load_state_dict(modelState)

    """
    Loading the training and validation. Also, it sets how the negative example will be generated.
    """
    tripletCollate = TripletBugCollate(inputHandlers)

    # load training
    if args.get('triplets_training'):
        negativePairGenOpt = args.get('neg_pair_generator', )
        tripletTrainingFile = args.get('triplets_training')

        offlineGeneration = not (negativePairGenOpt is None or negativePairGenOpt['type'] == 'none')

        if not offlineGeneration:
            logger.info("Not generate dynamically the negative examples.")
            tripletTrainingReader = TripletBugDatasetReader(tripletTrainingFile, preprocessors)
        else:
            pairGenType = negativePairGenOpt['type']
            masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

            if pairGenType == 'random':
                logger.info("Random Negative Pair Generator")
                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        trainingDataset.info, len(bugIds)))

                negativePairGenerator = RandomGenerator(preprocessors, tripletCollate,
                                                        negativePairGenOpt['rate'],
                                                        bugIds, masterIdByBugId)

            elif pairGenType == 'non_negative':
                logger.info("Non Negative Pair Generator")
                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        trainingDataset.info, len(bugIds)))

                negativePairGenerator = NonNegativeRandomGenerator(preprocessors, tripletCollate,
                                                                   negativePairGenOpt['rate'],
                                                                   bugIds, masterIdByBugId,
                                                                   negativePairGenOpt['n_tries'], device,
                                                                   decimals=negativePairGenOpt['decimals'])
            elif pairGenType == 'random_k':
                logger.info("Random K Negative Pair Generator")
                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        trainingDataset.info, len(bugIds)))

                negativePairGenerator = KRandomGenerator(preprocessors, tripletCollate,
                                                         negativePairGenOpt['rate'],
                                                         bugIds, masterIdByBugId, negativePairGenOpt['k'], device)
            elif pairGenType == "pre":
                logger.info("Pre-selected list generator")
                negativePairGenerator = PreSelectedGenerator(negativePairGenOpt['pre_list_file'], preprocessors,
                                                             negativePairGenOpt['rate'], masterIdByBugId,
                                                             negativePairGenOpt['preselected_length'])
            elif pairGenType == "misc_non_zero_pre":
                logger.info("Pre-selected list generator")

                negativePairGenerator1 = PreSelectedGenerator(negativePairGenOpt['pre_list_file'], preprocessors,
                                                              negativePairGenOpt['rate'], masterIdByBugId,
                                                              negativePairGenOpt['preselected_length'])

                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                negativePairGenerator2 = NonNegativeRandomGenerator(preprocessors, tripletCollate,
                                                                    negativePairGenOpt['rate'],
                                                                    bugIds, masterIdByBugId,
                                                                    negativePairGenOpt['n_tries'], device)

                negativePairGenerator = MiscOfflineGenerator((negativePairGenerator1, negativePairGenerator2))
            else:
                raise ArgumentError(
                    "Offline generator is invalid (%s). You should choose one of these: random, hard and pre" %
                    pairGenType)

            tripletTrainingReader = TripletBugDatasetReader(tripletTrainingFile, preprocessors, negativePairGenerator)

        trainingLoader = DataLoader(tripletTrainingReader, batch_size=batchSize, collate_fn=tripletCollate.collate,
                                    shuffle=True)
        logger.info("Training size: %s" % (len(trainingLoader.dataset)))

    # load validation
    if args.get('triplets_validation'):
        tripletValidationReader = TripletBugDatasetReader(args.get('triplets_validation'), preprocessors)
        validationLoader = DataLoader(tripletValidationReader, batch_size=batchSize, collate_fn=tripletCollate.collate)

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
    rankingScorer = SharedEncoderNNScorer(preprocessors, inputHandlers, model, device,
                                          batchSize=args['ranking_batch_size'])
    recallEstimationTrainOpt = args.get('recall_estimation_train')

    if recallEstimationTrainOpt:
        preselectListRankingTrain = PreselectListRanking(recallEstimationTrainOpt)

    recallEstimationOpt = args.get('recall_estimation')

    if recallEstimationOpt:
        preselectListRanking = PreselectListRanking(recallEstimationOpt)

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
        model.train()
        optimizer.zero_grad()
        x, y = tripletCollate.to(batch, device)
        output = model(*x)
        loss = lossFn(output, y)
        loss.backward()
        optimizer.step()
        return loss, output

    trainer = Engine(trainingIteration)
    trainingMetrics = {'training_loss': AverageLoss(lossFn, batch_size=lambda x: x.shape[0])}

    # Add metrics to trainer
    for name, metric in trainingMetrics.items():
        metric.attach(trainer, name)

    # Set validation functions
    def validationIteration(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = TripletBugCollate.to(batch, device)
            y_pred = model(*x)
            return y_pred, y_pred

    validationMetrics = {'validation_loss': LossWrapper(lossFn, batch_size=lambda x: x.shape[0])}
    evaluator = Engine(validationIteration)

    # Add metrics to evaluator
    for name, metric in validationMetrics.items():
        metric.attach(evaluator, name)

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
                             "train")

        if recallEstimationOpt and (epoch % args['rr_val_epoch'] == 0):
            logRankingResult(_run, logger, preselectListRanking, rankingScorer, bugReportDatabase,
                             args.get("ranking_result_file"), epoch, "validation")

        if offlineGeneration:
            tripletTrainingReader.sampleNewNegExamples(model, lossNoReduction)

        if args.get('save'):
            modelInfo = {'model': model.state_dict(),
                         'params': parametersToSave}

            logger.info("==> Saving Model: %s" % args['save'])
            torch.save(modelInfo, args['save'])

    if args.get('triplets_training'):
        trainer.run(trainingLoader, max_epochs=args['epochs'])
    elif args.get('triplets_validation'):
        # Evaluate Training
        evaluator.run(trainingLoader)
        logMetrics(logger, evaluator.state.metrics)

        if recallEstimationOpt:
            logRankingResult(_run, logger, preselectListRanking, rankingScorer, bugReportDatabase,
                             args.get("ranking_result_file"), 0, "validation")

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
                         recallRateOpt["result_file"], 0, None, group_by_master)
