"""
This script implements a siamese neural network which extract the information from each bug report of a pair.
The extracted features from each pair is used to calculate the similarity of them or probability of being duplicate.
"""
import codecs
import logging
import os
from argparse import ArgumentError
from datetime import datetime

import ignite
import numpy as np
import torch
from ignite.engine import Events, Engine
from nltk import WhitespaceTokenizer
from sacred import Experiment
from torch import optim
from torch.nn import NLLLoss, BCELoss
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR
from torch.utils.data import DataLoader

from data.Embedding import Embedding
from data.Lexicon import Lexicon
from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from data.collate import PairBugCollate
from data.dataset import PairBugDatasetReader
from data.input_handler import TextCNNInputHandler, DBRDCNN_CategoricalInputHandler
from data.preprocessing import PreprocessorList, CategoricalPreprocessor, DBR_CNN_CategoricalPreprocessor, \
    SummaryDescriptionPreprocessor, loadFilters, MultiLineTokenizer
from example_generator.offline_pair_geneneration import NonNegativeRandomGenerator, PreSelectedGenerator, \
    RandomGenerator, KRandomGenerator, MiscNonZeroRandomGen, MiscOfflineGenerator
from metrics.metric import AverageLoss, MeanScoreDistance, ConfusionMatrix, cmAccuracy, cmPrecision, cmRecall, \
    PredictionCache, LossWrapper, AccuracyWrapper, PrecisionWrapper, RecallWrapper
from metrics.ranking import PreselectListRanking, SharedEncoderNNScorer, \
    SunRanking, DeshmukhRanking, GeneralScorer, generateRecommendationList, DBR_CNN_Scorer
from model.dbr_cnn import DBR_CNN
from model.loss import CosineLoss, NeculoiuLoss
from model.siamese import ProbabilityPairNN, CosinePairNN
from util.jsontools import JsonLogFormatter
from util.siamese_util import processSumDescParam, processSumParam, processCategoricalParam, processDescriptionParam
from util.torch_util import thresholded_output_transform
from util.training_loop_util import logMetrics, logRankingResult, logConfusionMatrix

ex = Experiment("DBR-CNN")

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
    ranking_batch_size = 256
    cuda = True
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
        "random_anchor": False
    }
    dbr_cnn = {
        "categorical_lexicon": None,
        "word_embedding": None,
        "lexicon": None,
        "tokenizer": None,
        "filters": [],
        "window": [4, 5, 6],
        "nfilters": 100,
        "update_embedding": False
    }
    recall_estimation_train = None
    recall_estimation = None
    rr_val_epoch = 1
    rr_train_epoch = 5
    random_switch = False
    ranking_result_file = None
    optimizer = "adam"
    momentum = 0.9
    lr_scheduler = {
        "type": "constant",
        "decay": 1,
        "step_size": 1
    }
    save = None
    load = None
    pair_test_dataset = None

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
    # Setting logger
    args = _config
    logger = _log

    logger.info(args)
    logger.info('It started at: %s' % datetime.now())

    torch.manual_seed(_seed)

    device = torch.device('cuda' if args['cuda'] else "cpu")
    if args['cuda']:
        logger.info("Turning CUDA on")
    else:
        logger.info("Turning CUDA off")

    # Setting the parameter to save and loading parameters
    important_parameters = ['dbr_cnn']
    parameters_to_save = dict([(name, args[name]) for name in important_parameters])

    if args['load'] is not None:
        map_location = (lambda storage, loc: storage.cuda()) if args['cuda'] else 'cpu'
        model_info = torch.load(args['load'], map_location=map_location)
        model_state = model_info['model']

        for param_name, param_value in model_info['params'].items():
            args[param_name] = param_value
    else:
        model_state = None

    # Set basic variables
    preprocessors = PreprocessorList()
    input_handlers = []
    report_database = BugReportDatabase.fromJson(args['bug_database'])
    batchSize = args['batch_size']
    dbr_cnn_opt = args['dbr_cnn']

    # Loading word embedding and lexicon
    emb = np.load(dbr_cnn_opt["word_embedding"])
    padding_sym = "</s>"

    lexicon = Lexicon(unknownSymbol=None)
    with codecs.open(dbr_cnn_opt["lexicon"]) as f:
        for l in f:
            lexicon.put(l.strip())

    lexicon.setUnknown("UUUKNNN")
    padding_id = lexicon.getLexiconIndex(padding_sym)
    embedding = Embedding(lexicon, emb, paddingIdx=padding_id)

    logger.info("Lexicon size: %d" % (lexicon.getLen()))
    logger.info("Word Embedding size: %d" % (embedding.getEmbeddingSize()))

    # Load filters and tokenizer
    filters = loadFilters(dbr_cnn_opt['filters'])

    if dbr_cnn_opt['tokenizer'] == 'default':
        logger.info("Use default tokenizer to tokenize summary information")
        tokenizer = MultiLineTokenizer()
    elif dbr_cnn_opt['tokenizer'] == 'white_space':
        logger.info("Use white space tokenizer to tokenize summary information")
        tokenizer = WhitespaceTokenizer()
    else:
        raise ArgumentError(
            "Tokenizer value %s is invalid. You should choose one of these: default and white_space" %
            dbr_cnn_opt['tokenizer'])

    # Add preprocessors
    preprocessors.append(DBR_CNN_CategoricalPreprocessor(dbr_cnn_opt['categorical_lexicon'], report_database))
    preprocessors.append(SummaryDescriptionPreprocessor(lexicon, report_database, filters, tokenizer, padding_id))

    # Add input_handlers
    input_handlers.append(DBRDCNN_CategoricalInputHandler())
    input_handlers.append(TextCNNInputHandler(padding_id, min(dbr_cnn_opt["window"])))

    # Create Model
    model = DBR_CNN(embedding, dbr_cnn_opt["window"], dbr_cnn_opt["nfilters"], dbr_cnn_opt['update_embedding'])

    model.to(device)

    if model_state:
        model.load_state_dict(model_state)

    # Set loss function
    logger.info("Using BCE Loss")
    loss_fn = BCELoss()
    loss_no_reduction = BCELoss(reduction='none')
    cmp_collate = PairBugCollate(input_handlers, torch.float32, unsqueeze_target=True)

    # Loading the training and setting how the negative example will be generated.
    if args.get('pairs_training'):
        negative_pair_gen_opt = args.get('neg_pair_generator', )
        pairsTrainingFile = args.get('pairs_training')
        random_anchor = negative_pair_gen_opt['random_anchor']

        offlineGeneration = not (negative_pair_gen_opt is None or negative_pair_gen_opt['type'] == 'none')

        if not offlineGeneration:
            logger.info("Not generate dynamically the negative examples.")
            pair_training_reader = PairBugDatasetReader(pairsTrainingFile, preprocessors,
                                                        randomInvertPair=args['random_switch'])
        else:
            pair_gen_type = negative_pair_gen_opt['type']
            master_id_by_bug_id = report_database.getMasterIdByBugId()

            if pair_gen_type == 'random':
                logger.info("Random Negative Pair Generator")
                training_dataset = BugDataset(negative_pair_gen_opt['training'])
                bug_ids = training_dataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        training_dataset.info, len(bug_ids)))

                negative_pair_generator = RandomGenerator(preprocessors, cmp_collate,
                                                          negative_pair_gen_opt['rate'],
                                                          bug_ids, master_id_by_bug_id)

            elif pair_gen_type == 'non_negative':
                logger.info("Non Negative Pair Generator")
                training_dataset = BugDataset(negative_pair_gen_opt['training'])
                bug_ids = training_dataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        training_dataset.info, len(bug_ids)))

                negative_pair_generator = NonNegativeRandomGenerator(preprocessors, cmp_collate,
                                                                   negative_pair_gen_opt['rate'],
                                                                   bug_ids, master_id_by_bug_id,
                                                                   negative_pair_gen_opt['n_tries'],
                                                                   device,
                                                                   randomAnchor=random_anchor)
            elif pair_gen_type == 'misc_non_zero':
                logger.info("Misc Non Zero Pair Generator")
                training_dataset = BugDataset(negative_pair_gen_opt['training'])
                bug_ids = training_dataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        training_dataset.info, len(bug_ids)))

                negative_pair_generator = MiscNonZeroRandomGen(preprocessors, cmp_collate,
                                                               negative_pair_gen_opt['rate'], bug_ids,
                                                               training_dataset.duplicateIds, master_id_by_bug_id,
                                                               device,
                                                               negative_pair_gen_opt['n_tries'],
                                                               negative_pair_gen_opt['random_anchor'])
            elif pair_gen_type == 'random_k':
                logger.info("Random K Negative Pair Generator")
                training_dataset = BugDataset(negative_pair_gen_opt['training'])
                bug_ids = training_dataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        training_dataset.info, len(bug_ids)))

                negative_pair_generator = KRandomGenerator(preprocessors, cmp_collate, negative_pair_gen_opt['rate'],
                                                           bug_ids, master_id_by_bug_id, negative_pair_gen_opt['k'],
                                                           device)
            elif pair_gen_type == "pre":
                logger.info("Pre-selected list generator")
                negative_pair_generator = PreSelectedGenerator(negative_pair_gen_opt['pre_list_file'], preprocessors,
                                                               negative_pair_gen_opt['rate'], master_id_by_bug_id,
                                                               negative_pair_gen_opt['preselected_length'])
            elif pair_gen_type == "misc_non_zero_pre":
                logger.info("Pre-selected list generator")

                negativePairGenerator1 = PreSelectedGenerator(negative_pair_gen_opt['pre_list_file'], preprocessors,
                                                              negative_pair_gen_opt['rate'], master_id_by_bug_id,
                                                              negative_pair_gen_opt['preselected_length'])

                training_dataset = BugDataset(negative_pair_gen_opt['training'])
                bug_ids = training_dataset.bugIds

                negativePairGenerator2 = NonNegativeRandomGenerator(preprocessors, cmp_collate,
                                                                    negative_pair_gen_opt['rate'],
                                                                    bug_ids, master_id_by_bug_id, device,
                                                                    negative_pair_gen_opt['n_tries'])

                negative_pair_generator = MiscOfflineGenerator((negativePairGenerator1, negativePairGenerator2))
            else:
                raise ArgumentError(
                    "Offline generator is invalid (%s). You should choose one of these: random, hard and pre" %
                    pair_gen_type)

            pair_training_reader = PairBugDatasetReader(pairsTrainingFile, preprocessors, negative_pair_generator,
                                                        randomInvertPair=args['random_switch'])

        training_loader = DataLoader(pair_training_reader, batch_size=batchSize, collate_fn=cmp_collate.collate,
                                     shuffle=True)
        logger.info("Training size: %s" % (len(training_loader.dataset)))

    # load validation
    if args.get('pairs_validation'):
        pair_validation_reader = PairBugDatasetReader(args.get('pairs_validation'), preprocessors)
        validation_loader = DataLoader(pair_validation_reader, batch_size=batchSize, collate_fn=cmp_collate.collate)

        logger.info("Validation size: %s" % (len(validation_loader.dataset)))
    else:
        validation_loader = None

    """
    Training and evaluate the model. 
    """
    optimizer_opt = args.get('optimizer', 'adam')

    if optimizer_opt == 'sgd':
        logger.info('SGD')
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['l2'], momentum=args['momentum'])
    elif optimizer_opt == 'adam':
        logger.info('Adam')
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2'])

    # Recall rate
    ranking_scorer = DBR_CNN_Scorer(preprocessors[0], preprocessors[1], input_handlers[0], input_handlers[1], model,
                                    device, args['ranking_batch_size'])
    recallEstimationTrainOpt = args.get('recall_estimation_train')

    if recallEstimationTrainOpt:
        preselectListRankingTrain = PreselectListRanking(recallEstimationTrainOpt)

    recallEstimationOpt = args.get('recall_estimation')

    if recallEstimationOpt:
        preselect_list_ranking = PreselectListRanking(recallEstimationOpt)

    lr_scheduler_opt = args.get('lr_scheduler', None)

    if lr_scheduler_opt is None or lr_scheduler_opt['type'] == 'constant':
        logger.info("Scheduler: Constant")
        lr_sched = None
    elif lr_scheduler_opt["type"] == 'step':
        logger.info("Scheduler: StepLR (step:%s, decay:%f)" % (lr_scheduler_opt["step_size"], args["decay"]))
        lr_sched = StepLR(optimizer, lr_scheduler_opt["step_size"], lr_scheduler_opt["decay"])
    elif lr_scheduler_opt["type"] == 'exp':
        logger.info("Scheduler: ExponentialLR (decay:%f)" % (lr_scheduler_opt["decay"]))
        lr_sched = ExponentialLR(optimizer, lr_scheduler_opt["decay"])
    elif lr_scheduler_opt["type"] == 'linear':
        logger.info("Scheduler: Divide by (1 + epoch * decay) ---- (decay:%f)" % (lr_scheduler_opt["decay"]))

        lrDecay = lr_scheduler_opt["decay"]
        lr_sched = LambdaLR(optimizer, lambda epoch: 1 / (1.0 + epoch * lrDecay))
    else:
        raise ArgumentError(
            "LR Scheduler is invalid (%s). You should choose one of these: step, exp and linear " %
            pair_gen_type)

    # Set training functions
    def trainingIteration(engine, batch):
        model.train()

        optimizer.zero_grad()
        x, y = cmp_collate.to(batch, device)
        output = model(*x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        return loss, output, y

    trainer = Engine(trainingIteration)
    negTarget = 0.0 if isinstance(loss_fn, NLLLoss) else -1.0

    trainingMetrics = {'training_loss': AverageLoss(loss_fn),
                       'training_acc': AccuracyWrapper(output_transform=thresholded_output_transform),
                       'training_precision': PrecisionWrapper(output_transform=thresholded_output_transform),
                       'training_recall': RecallWrapper(output_transform=thresholded_output_transform),
                       }

    # Add metrics to trainer
    for name, metric in trainingMetrics.items():
        metric.attach(trainer, name)

    # Set validation functions
    def validationIteration(engine, batch):
        model.eval()

        with torch.no_grad():
            x, y = cmp_collate.to(batch, device)
            y_pred = model(*x)

            return y_pred, y

    validationMetrics = {'validation_loss': LossWrapper(loss_fn),
                         'validation_acc': AccuracyWrapper(output_transform=thresholded_output_transform),
                         'validation_precision': PrecisionWrapper(output_transform=thresholded_output_transform),
                         'validation_recall': RecallWrapper(output_transform=thresholded_output_transform),
                         }
    evaluator = Engine(validationIteration)

    # Add metrics to evaluator
    for name, metric in validationMetrics.items():
        metric.attach(evaluator, name)

    @trainer.on(Events.EPOCH_STARTED)
    def onStartEpoch(engine):
        epoch = engine.state.epoch
        logger.info("Epoch: %d" % epoch)

        if lr_sched:
            lr_sched.step()

        logger.info("LR: %s" % str(optimizer.param_groups[0]["lr"]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def onEndEpoch(engine):
        epoch = engine.state.epoch

        logMetrics(_run, logger, engine.state.metrics, epoch)

        # Evaluate Training
        if validation_loader:
            evaluator.run(validation_loader)
            logMetrics(_run, logger, evaluator.state.metrics, epoch)

        lastEpoch = args['epochs'] - epoch == 0

        if recallEstimationTrainOpt and (epoch % args['rr_train_epoch'] == 0):
            logRankingResult(_run, logger, preselectListRankingTrain, ranking_scorer, report_database, None, epoch,
                             "train")
            ranking_scorer.free()

        if recallEstimationOpt and (epoch % args['rr_val_epoch'] == 0):
            logRankingResult(_run, logger, preselect_list_ranking, ranking_scorer, report_database,
                             args.get("ranking_result_file"), epoch, "validation")
            ranking_scorer.free()

        if not lastEpoch:
            pair_training_reader.sampleNewNegExamples(model, loss_no_reduction)

        if args.get('save'):
            save_by_epoch = args['save_by_epoch']

            if save_by_epoch and epoch in save_by_epoch:
                file_name, file_extension = os.path.splitext(args['save'])
                file_path = file_name + '_epoch_{}'.format(epoch) + file_extension
            else:
                file_path = args['save']

            modelInfo = {'model': model.state_dict(),
                         'params': parameters_to_save}

            logger.info("==> Saving Model: %s" % file_path)
            torch.save(modelInfo, file_path)

    if args.get('pairs_training'):
        trainer.run(training_loader, max_epochs=args['epochs'])
    elif args.get('pairs_validation'):
        # Evaluate Training
        evaluator.run(validation_loader)
        logMetrics(logger, evaluator.state.metrics)

        if recallEstimationOpt:
            logRankingResult(_run, logger, preselect_list_ranking, ranking_scorer, report_database,
                             args.get("ranking_result_file"), 0, "validation")

    # Test Dataset (accuracy, recall, precision, F1)
    pair_test_dataset = args.get('pair_test_dataset')

    if pair_test_dataset is not None and len(pair_test_dataset) > 0:
        pairTestReader = PairBugDatasetReader(pair_test_dataset, preprocessors)
        testLoader = DataLoader(pairTestReader, batch_size=batchSize, collate_fn=cmp_collate.collate)

        if not isinstance(cmp_collate, PairBugCollate):
            raise NotImplementedError('Evaluation of pairs using tanh was not implemented yet')

        logger.info("Test size: %s" % (len(testLoader.dataset)))

        testMetrics = {'test_accuracy': ignite.metrics.Accuracy(output_transform=thresholded_output_transform),
                       'test_precision': ignite.metrics.Precision(output_transform=thresholded_output_transform),
                       'test_recall': ignite.metrics.Recall(output_transform=thresholded_output_transform),
                       'test_predictions': PredictionCache(), }
        test_evaluator = Engine(validationIteration)

        # Add metrics to evaluator
        for name, metric in testMetrics.items():
            metric.attach(test_evaluator, name)

        test_evaluator.run(testLoader)

        for metricName, metricValue in test_evaluator.state.metrics.items():
            metric = testMetrics[metricName]

            if isinstance(metric, ignite.metrics.Accuracy):
                logger.info({'type': 'metric', 'label': metricName, 'value': metricValue, 'epoch': None,
                             'correct': metric._num_correct, 'total': metric._num_examples})
                _run.log_scalar(metricName, metricValue)
            elif isinstance(metric, (ignite.metrics.Precision, ignite.metrics.Recall)):
                logger.info({'type': 'metric', 'label': metricName, 'value': metricValue,
                             'epoch': None,
                             'tp': metric._true_positives.item(),
                             'total_positive': metric._positives.item()
                             })
                _run.log_scalar(metricName, metricValue)
            elif isinstance(metric, ConfusionMatrix):
                acc = cmAccuracy(metricValue)
                prec = cmPrecision(metricValue, False)
                recall = cmRecall(metricValue, False)
                f1 = 2 * (prec * recall) / (prec + recall + 1e-15)

                logger.info(
                    {'type': 'metric', 'label': metricName,
                     'accuracy': np.float(acc),
                     'precision': prec.cpu().numpy().tolist(),
                     'recall': recall.cpu().numpy().tolist(),
                     'f1': f1.cpu().numpy().tolist(),
                     'confusion_matrix': metricValue.cpu().numpy().tolist(),
                     'epoch': None})

                _run.log_scalar('test_f1', f1[1])
            elif isinstance(metric, PredictionCache):
                logger.info(
                    {'type': 'metric', 'label': metricName,
                     'predictions': metric.predictions})

    # Calculate recall rate
    recall_rate_opt = args.get('recall_rate', {'type': 'none'})
    if recall_rate_opt['type'] != 'none':
        if recall_rate_opt['type'] == 'sun2011':
            logger.info("Calculating recall rate: {}".format(recall_rate_opt['type']))
            recall_rate_dataset = BugDataset(recall_rate_opt['dataset'])

            ranking_class = SunRanking(report_database, recall_rate_dataset, recall_rate_opt['window'])
            # We always group all bug reports by master in the results in the sun 2011 methodology
            group_by_master = True
        elif recall_rate_opt['type'] == 'deshmukh':
            logger.info("Calculating recall rate: {}".format(recall_rate_opt['type']))
            recall_rate_dataset = BugDataset(recall_rate_opt['dataset'])
            ranking_class = DeshmukhRanking(report_database, recall_rate_dataset)
            group_by_master = recall_rate_opt['group_by_master']
        else:
            raise ArgumentError(
                "recall_rate.type is invalid (%s). You should choose one of these: step, exp and linear " %
                recall_rate_opt['type'])

        logRankingResult(_run, logger, ranking_class, ranking_scorer, report_database,
                         recall_rate_opt["result_file"], 0, None, group_by_master, )
