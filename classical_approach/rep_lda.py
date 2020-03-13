import argparse
import codecs
import logging
import math
import os
import pickle
import random
import re
import subprocess

import numpy as np
import gensim
from gensim import utils
from gensim.models import ldamulticore
from scipy.spatial.distance import jensenshannon
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import LeaveOneOut, KFold

import classical_approach.generate_input_dbrd as dbrd_in
import os.path as path

from gensim.matutils import (
    kullback_leibler, hellinger, jaccard_distance, jensen_shannon,
    dirichlet_expectation, logsumexp, mean_absolute_difference
)
from sklearn.decomposition import LatentDirichletAllocation

# ./fast-dbrd -n rep_full_2001-2007_365_validation -r /home/irving/fast-dbrd/ranknet-configs/full-textual-full-categorial.cfg --training-duplicates 27481 --ts  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/timestamp_file.txt --time-constraint 365 --recommend  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/dbrd_validation.txt
from classical_approach.bm25f import SUN_REPORT_ID_INDEX
from classical_approach.dbtm import Tmodel
from classical_approach.generate_input_dbrd import DBRDPreprocessing, generate_input_vec
from classical_approach.read_data import read_weights, read_dbrd_file
from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from metrics.ranking import SunRanking
from util.training_loop_util import logRankingResult


def lda_similarity(q, c):
    sim = jensenshannon(q, c)

    if np.isnan(sim):
        if np.absolute(q-c).sum() < 0.000001:
            sim = 0
        else:
            raise Exception("jensen-shannon similairty is nan")

    return sim

class REP_LDA_Scorer:

    def __init__(self, bow, lda, rep):
        self.bow = bow
        self.lda = lda
        self.rep = rep

    def pregenerateBugEmbedding(self, allBugIds):
        pass

    def score(self, anchorBugId, bugIds):
        query_rep_inp = self.topic_by_id[anchorBugId]
        similarities = []
        for cand_id in bugIds:
            cand_rep_inp = self.topic_by_id[cand_id]

            similarities.append(lda_similarity(query_rep_inp, cand_rep_inp))

        return similarities

    def reset(self):
        pass

    def free(self):
        pass


class LDAScorer(object):
    def __init__(self, lda, topic_by_id):
        self.topic_by_id = topic_by_id
        self.lda = lda

    def pregenerateBugEmbedding(self, allBugIds):
        pass

    def score(self, anchorBugId, bugIds):
        query_rep_inp = self.topic_by_id[anchorBugId]
        similarities = []
        for cand_id in bugIds:
            cand_rep_inp = self.topic_by_id[cand_id]

            similarities.append(lda_similarity(query_rep_inp, cand_rep_inp))

        return similarities

    def reset(self):
        pass

    def free(self):
        pass


class REP_LDA_Scorer(object):
    def __init__(self, rep, a1, a2, dbrd_input_by_id, topic_dist_by_id):
        self.dbrd_input_by_id = dbrd_input_by_id
        self.topic_dist_by_id = topic_dist_by_id
        self.a1 = a1
        self.a2 = a2
        self.rep = rep

    def pregenerateBugEmbedding(self, allBugIds):
        pass

    def score(self, anchorBugId, bugIds):
        query_rep_inp = self.dbrd_input_by_id[anchorBugId]
        similarities = []
        for cand_id in bugIds:
            cand_rep_inp = self.dbrd_input_by_id[cand_id]

            rep_sim = self.rep.similarity(query_rep_inp, cand_rep_inp)
            lda_sim = lda_similarity(self.topic_dist_by_id[anchorBugId], self.topic_dist_by_id[cand_id])
            sim = self.a1 * rep_sim + self.a2 * lda_sim

            similarities.append(sim)

        return similarities

    def reset(self):
        pass

    def free(self):
        pass


def generate_cand_list(test_id, masterSetById, database, list_size=10000):
    test = database.getBug(test_id)
    master = masterSetById[test['dup_id']]

    candidates = []

    for cand in master:
        if int(cand) >= int(test['bug_id']):
            continue

        candidates.append(cand)

    n = 0
    current_id = int(test_id)

    while n < list_size:
        current_id -= 1

        if current_id == 0:
            break

        cand_id = str(current_id)
        cand = database.bugById.get(cand_id)

        if not cand:
            continue

        if cand_id in master:
            continue

        candidates.append(cand['bug_id'])

        n += 1

    return candidates


def tune_alpha(bow, rep, lda, query_idxs, masterSetById, database, n_candidates, dbrd_input_by_id, lda_dist_by_id):
    similarities = []
    queries = []

    logger = logging.getLogger()

    logger.info("Tuning alpha")


    for idx, query_id in enumerate(query_idxs):
        candidates = generate_cand_list(query_id, masterSetById, database, n_candidates)
        query = database.getBug(query_id)
        query_id = query['bug_id']
        dup_id = query['dup_id']

        topic_dist_report = lda.transform(bow.transform(join_sum_desc([query])))[0]

        query_input = dbrd_input_by_id[query_id]
        rep_sim_list = [rep.similarity(query_input, dbrd_input_by_id[cand]) for cand in candidates]
        lda_sim_list = [lda_similarity(topic_dist_report, lda_dist_by_id[cand]) for cand in candidates]

        similarities.append(list(zip(candidates, rep_sim_list, lda_sim_list)))
        queries.append((query_id, dup_id))

    masterIdByBugId = database.getMasterIdByBugId()
    map_results = []

    for a1 in np.arange(0., 1.01, 0.01):
        a2 = 1 - a1
        map_metric = 0

        for (query_id, dup_id), sim in zip(queries, similarities):
            masterSet = masterSetById[dup_id]

            recommendationList = sorted(map(lambda s: (s[0], a1 * s[1] + a2 * s[2]), sim), key=lambda v: v[1],
                                        reverse=True)
            seenMasters = set()

            for bugId, p in recommendationList:
                mastersetId = masterIdByBugId[bugId]

                if mastersetId in seenMasters:
                    continue

                seenMasters.add(mastersetId)

                if bugId in masterSet:
                    pos = len(seenMasters)
                    break

            map_metric += 1 / pos

        map_results.append([a1, map_metric])

    return map_results


class Run(object):

    def log_scalar(self, label, rate, step):
        pass


def filter_bugs(bugs, max_bug_id, test=set()):
    for bug in bugs:
        if int(bug['bug_id']) > max_bug_id or bug['bug_id'] in test:
            continue

        yield bug


def join_sum_desc(filtered_bugs):
    for bug in filtered_bugs:
        yield bug['short_desc'] + '\n' + bug['description'] + '\n PPPAAADDD'


def train_dbtm(bugReportDatabase, base_filename, test, max_bug_id, rep_reports=None):
    input_file = path.join(args.dir, "{}_{}.txt".format(base_filename, "drbd_input"))

    # Generate file to train REP
    if rep_reports:
        dbrd_input, max_token_id = rep_reports
    else:
        dbrd_input, max_token_id = generate_input_vec(bugReportDatabase, max_bug_id)

    dbrd_in.generate_file(dbrd_input, input_file, set(test))

    # Count the number of duplicate reports in input_file
    n_dup_reports = 0
    regex = re.compile(r"DID=[0-9]+")
    with codecs.open(input_file) as dbrd_file:
        for l in dbrd_file:
            if regex.search(l):
                n_dup_reports += 1

    # Trainig REP
    # ./fast-dbrd -n rep_full_2001-2007_365_validation -r /home/
    # irving/fast-dbrd/ranknet-configs/full-textual-full-categorial.cfg --training-duplicates 27481 --ts  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/timestamp_file.txt --time-constraint 365 --recommend  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/dbrd_validation.txt
    logger.info([args.exe, "-n", args.l, "-r", args.conf, "--training-duplicates", str(n_dup_reports), input_file])
    sp = subprocess.Popen(
        [args.exe, "-n", args.l, "-r", args.conf, "--training-duplicates", str(n_dup_reports), input_file],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=args.dir)

    out, err = sp.communicate()

    logger.info(out)
    logger.info(err)

    # Load REP
    rep = read_weights(os.path.join(args.dir, "dbrd_ranknet_{}_I-1".format(args.l)))

    # REP - transform input and compute idf and mean length
    dbrd_input = rep.fit_transform(dbrd_input, max_token_id, False)
    dbrd_input_by_id = {}
    for inp in dbrd_input:
        dbrd_input_by_id[inp[SUN_REPORT_ID_INDEX]] = inp

    # Train LDA
    # ./fast-dbrd -n rep_full_2001-2007_365_validation -r /home/irving/fast-dbrd/ranknet-configs/full-textual-full-categorial.cfg --training-duplicates 27481 --ts  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/timestamp_file.txt --time-constraint 365 --recommend  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/dbrd_validation.txt
    bow = CountVectorizer(tokenizer=preprocessing.preprocess, stop_words=None, lowercase=True, analyzer='word')

    filtered_bugs = filter_bugs(bugReportDatabase.bugList, max_bug_id, test)
    merge_sum_desc = join_sum_desc(filtered_bugs)

#    X = bow.fit_transform(merge_sum_desc)
    lda = LatentDirichletAllocation(args.k, args.a, args.B, max_iter=args.it)

#    lda_out = lda.fit_transform(X)
#    logger.info("Lda output training")
#    logger.info(lda_out[:5])

    lda_dist_by_id = {}
#    for bug, _in in zip(filter_bugs(bugReportDatabase.bugList, max_bug_id, test), lda_out):
#        lda_dist_by_id[bug['bug_id']] = _in

    # Encode test
#    bugs = [bugReportDatabase.getBug(bug_id) for bug_id in test]

#    if len(bugs) > 0:
#        filtered_bugs = filter_bugs(bugs, max_bug_id)
#        merge_sum_desc = join_sum_desc(filtered_bugs)

#        X = bow.transform(merge_sum_desc)
#        lda_out = lda.transform(X)

#        for bug, _in in zip(bugs, lda_out):
#            lda_dist_by_id[bug['bug_id']] = _in

    return bow, lda, rep, dbrd_input_by_id, lda_dist_by_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-db", "--database", dest='db', required=True, help="")
    parser.add_argument("-dt", '--dataset', dest='dt', help="")
    parser.add_argument("-m", "--model", dest='m', required=True, help="")

    parser.add_argument("-exe", "--fast_dbrd", dest='exe', help="")
    parser.add_argument("-dir", "--directory", dest='dir', help="")

    parser.add_argument("-conf", "--rep_config", dest='conf', help="")
    parser.add_argument("-l", "--label", dest='l', default="dbtm", help="")

    parser.add_argument("-fold", "--fold", dest='fold', type=int, default=0, help="")

    parser.add_argument("-it", "--iterations", dest='it', type=int, help="")
    parser.add_argument("-k", "--n_topics", dest='k', type=int, help="")
    parser.add_argument("-a", "--alpha", dest='a', type=float, default=None, help="")
    parser.add_argument("-B", "--beta", dest='B', type=float, default=None, help="")

    parser.add_argument("-nc", "--n_cand", dest='nc', type=int, help="")

    parser.add_argument("--seed", default=554154, help="")

    parser.add_argument("-test", '--test', dest='test', help="")
    parser.add_argument("--result_file", help="")
    parser.add_argument("--window", type=int, help="")

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)

    training = True

    bugReportDatabase = BugReportDatabase.fromJson(args.db)
    testDataset = BugDataset(args.test)
    preprocessing = DBRDPreprocessing()

    if args.dt is not None:
        trainingDataset = BugDataset(args.dt)

        np.random.seed(args.seed)
        random.seed(args.seed)

        data_model = []
        data_alpha = []

        # Split the dataset into two sets: the first set is used to train the model and
        # the second is used to tune the ensemble weights
        duplicate_reports = trainingDataset.duplicateIds

        if args.fold < 0:
            logger.info("Validation set")
            random.shuffle(duplicate_reports)

            sample_size = int(len(duplicate_reports) * 0.05)
            data_alpha = sorted(duplicate_reports[:sample_size], key=lambda bug_id: int(bug_id))

            splits = [(sorted(duplicate_reports[sample_size:], key=lambda bug_id: int(bug_id)), data_alpha)]
        elif args.fold == 0:
            logger.info("Leave one out")
            loo = LeaveOneOut()

            splits = loo.split(duplicate_reports)
        else:
            logger.info("K-folds: {}".format(args.fold))
            kf = KFold(n_splits=args.fold)

            splits = kf.split(duplicate_reports)

        base_filename = path.splitext(path.split(args.dt)[1])[0] + '_' + args.l
        max_bug_id = max(map(lambda bug_id: int(bug_id), trainingDataset.bugIds))
        masterSetById = bugReportDatabase.getMasterSetById(trainingDataset.bugIds)
        map_by_alpha = []
        n_queries = 0
        # Preprocess the reports before writing REP input file
        rep_reports = generate_input_vec(bugReportDatabase, max_bug_id)

        for train_idx, test_idx in splits:
            break
            test = [duplicate_reports[idx] for idx in test_idx]
            bow, rep, lda, dbrd_input_by_id, lda_out_by_id = train_dbtm(bugReportDatabase, base_filename, test,
                                                                        max_bug_id, rep_reports)

            # Tune ensemble parameters
            map_results = tune_alpha(bow, lda, rep, test, masterSetById, bugReportDatabase, args.nc, dbrd_input_by_id,
                                     lda_out_by_id)
            n_queries += len(test)

            if len(map_by_alpha) > 0:
                for idx in range(len(map_results)):
                    map_by_alpha[idx][1] += map_results[idx][1]
            else:
                map_by_alpha = map_results

#        map_by_alpha = sorted(map_by_alpha, key=lambda k: k[1], reverse=True)

        a1 = 1
        a2 = 1 - a1

#        logger.info("Best MAP: {}".format(map_by_alpha[:10]))
#        logger.info("Worst MAP: {}".format(map_by_alpha[-10:]))
#        logger.info("a1={} and a2={} achieved the best MAP({}). Number of queries={}".format(a1, a2, map_by_alpha[0],
#                                                                                             n_queries))

        bow, lda, rep, dbrd_input_by_id, lda_out_by_id = train_dbtm(bugReportDatabase, base_filename, [], max_bug_id,
                                                                    rep_reports)

        to_save = (bow, rep, lda, a1, a2)
        # Save Model
        pickle.dump(to_save, open(args.m, "wb"))

    if args.test is not None:
        bugReportDatabase = BugReportDatabase.fromJson(args.db)

        bow, rep, lda, a1, a2 = pickle.load(open(args.m, "rb"))

        testDataset = BugDataset(args.test)
        rankingClass = SunRanking(bugReportDatabase, testDataset, args.window)
        group_by_master = True

        # Generate REP inputs
        max_bug_id = max(map(lambda bug_id: int(bug_id), testDataset.bugIds))
        dbrd_input, max_token_id = generate_input_vec(bugReportDatabase, max_bug_id)
        rep.fit_transform(dbrd_input, max_token_id, True)

        dbrd_input_by_id = {}

        for inp in dbrd_input:
            dbrd_input_by_id[inp[SUN_REPORT_ID_INDEX]] = inp

        # Estimate the topic proportion of each new report
        filtered_bugs = filter_bugs(bugReportDatabase.bugList, max_bug_id)
        X = bow.fit_transform(join_sum_desc(filtered_bugs))

        topics_document = lda.fit_transform(X)

#        logger.info("Lda output test")
#        logger.info(topics_document[:5])

        topic_by_bug_id = {}

        for bug, topic_dist in zip(filter_bugs(bugReportDatabase.bugList, max_bug_id), topics_document):
            topic_by_bug_id[bug['bug_id']] = topic_dist

        # Evaluate method
        for a1 in np.arange(0.1, 1.01, 0.1):
            a2 = 1 - a1
            logger.info("a1={} and a2={}".format(a1,a2))
            rankingScorer = REP_LDA_Scorer(rep, a1, a2, dbrd_input_by_id, topic_by_bug_id)
            rankingClass = SunRanking(bugReportDatabase, testDataset, args.window)
            group_by_master = True

            logRankingResult(Run(), logger, rankingClass, rankingScorer, bugReportDatabase,
                         args.result_file, 0, None, group_by_master)

