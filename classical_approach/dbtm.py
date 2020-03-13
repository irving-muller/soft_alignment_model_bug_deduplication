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
from sklearn.model_selection import LeaveOneOut, KFold

import classical_approach.generate_input_dbrd as dbrd_in
import os.path as path

from gensim.matutils import (
    kullback_leibler, hellinger, jaccard_distance, jensen_shannon,
    dirichlet_expectation, logsumexp, mean_absolute_difference
)
from sklearn.decomposition import LatentDirichletAllocation

# http://www.acme.byu.edu/wp-content/uploads/2018/02/GibbsLDA.pdf
# http://brooksandrew.github.io/simpleblog/articles/latent-dirichlet-allocation-under-the-hood/
# http://173.236.226.255/tom/papers/SteyversGriffiths.pdf?source=post_page---------------------------
# JGibbsLDA

# ./fast-dbrd -n rep_full_2001-2007_365_validation -r /home/irving/fast-dbrd/ranknet-configs/full-textual-full-categorial.cfg --training-duplicates 27481 --ts  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/timestamp_file.txt --time-constraint 365 --recommend  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/dbrd_validation.txt
from classical_approach.bm25f import SUN_REPORT_ID_INDEX
from classical_approach.generate_input_dbrd import DBRDPreprocessing, generate_input_vec
from classical_approach.read_data import read_weights, read_dbrd_file
from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from metrics.ranking import SunRanking
from util.training_loop_util import logRankingResult


class Tmodel(object):

    def __init__(self, n_topics, alpha, beta):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

        self.preprocessing = DBRDPreprocessing()

    def initialize(self, corpus):
        """
        Initialize the matrices and some variables
        :param corpus:
        :return:
        """
        self.n_docs = len(corpus)
        self.vocab_size = len(self.vocab)
        self.matrix_doc_topic = np.zeros((self.n_docs, self.n_topics))
        self.matrix_word_topic = np.zeros((self.vocab_size, self.n_topics))
        self.master_ids = {}

        # Documents from the same master share the same array of topics
        self.matrix_master_topic = []

        pos_master = {}
        for bug_id, dup_group_id, words in corpus:
            if len(dup_group_id) == 0:
                dup_group_id = "NONE"

            vec = pos_master.setdefault(dup_group_id, np.zeros(self.n_topics))
            self.matrix_master_topic.append(vec)

        # Randomly sample the topic of each word in the documents
        self.topic_per_docs = []
        self.idx_bug = {}

        for doc_idx, (bug_id, dup_group_id, words) in enumerate(corpus):
            topics = np.random.randint(self.n_topics, size=len(words))

            for word, topic_idx in zip(words, topics):
                self.matrix_doc_topic[doc_idx][topic_idx] += 1
                self.matrix_word_topic[word][topic_idx] += 1
                self.matrix_master_topic[doc_idx][topic_idx] += 1

            self.idx_bug[bug_id] = doc_idx
            self.topic_per_docs.append(topics)

    def transform(self, raw_reports):
        reports = []
        for bug in raw_reports:
            sum = bug['short_desc']
            desc = bug['description']

            preprocessed_sum = self.preprocessing.preprocess(sum)
            preprocessed_desc = self.preprocessing.preprocess(desc)

            all_tokens = []

            for token in preprocessed_sum:
                token_id = self.vocab.get(token)

                if token_id is not None:
                    all_tokens.append(token_id)

            for token in preprocessed_desc:
                token_id = self.vocab.get(token)

                if token_id is not None:
                    all_tokens.append(token_id)

            reports.append((bug["bug_id"], all_tokens))

        return reports

    @staticmethod
    def preprocess(raw_reports, max_bug_id, black_list, mastersetByBugId):
        """

        :param database:
        :param max_bug_id:
        :param black_list:
        :return:
        """
        preprocessing = DBRDPreprocessing()
        token_freq = {}

        reports = []
        for bug in raw_reports:
            bug_id = bug['bug_id']

            if int(bug_id) > max_bug_id:
                continue

            if len(black_list) > 0 and bug_id in black_list:
                continue

            sum = bug['short_desc']
            desc = bug['description']
            dup_id = bug['dup_id']

            preprocessed_sum = preprocessing.preprocess(sum)
            preprocessed_desc = preprocessing.preprocess(desc)

            all_tokens = preprocessed_sum + preprocessed_desc

            for token in all_tokens:
                p = token_freq.setdefault(token, [len(token_freq), 0])
                p[1] += 1

            if len(dup_id) == 0:
                master = mastersetByBugId.get(bug_id)

                if master:
                    if len(master - black_list) > 1:
                        dup_id = bug_id

            reports.append([str(bug_id), dup_id, all_tokens])

        vocab = {}

        for k, v in token_freq.items():
            _, freq = v

            if freq == 1:
                continue

            vocab[k] = len(vocab)

        for report in reports:
            all_tokens = report[-1]
            all_token_ids = []

            for i in range(len(all_tokens)):
                token_id = vocab.get(all_tokens[i])

                if token_id is not None:
                    all_token_ids.append(token_id)

            report[-1] = all_token_ids

        return vocab, reports

    def compute_theta(self, matrix_doc_topic):
        return (matrix_doc_topic + self.alpha) / (
                matrix_doc_topic.sum(axis=1, keepdims=True) + self.alpha * self.n_topics)

    def compute_phi(self, matrix_word_topic):
        return (matrix_word_topic + self.beta) / (
                matrix_word_topic.sum(axis=0, keepdims=True) + self.vocab_size * self.beta)

    def fit(self, vocab, corpus, iteration):
        self.vocab = vocab
        self.initialize(corpus)

        # Removing self from the loop increases the code speed
        vocab_size = self.vocab_size
        matrix_master_topic = self.matrix_master_topic
        matrix_doc_topic = self.matrix_doc_topic
        matrix_word_topic = self.matrix_word_topic
        topic_per_docs = self.topic_per_docs
        alpha = self.alpha
        beta = self.beta
        n_topics = self.n_topics

        # Inference the parameters. Gibbs sampling
        topic_idxs = np.arange(n_topics)
        self.theta = self.compute_theta(matrix_doc_topic)

        logger = logging.getLogger()

        # TR
        for it in range(iteration):
            for doc_idx, (bug_id, dup_group_id, words) in enumerate(corpus):
                doc_len = len(words) - 1

                if len(dup_group_id) > 0:
                    # the sum of the lengths of all reports in duplicate group
                    dup_group_len = matrix_master_topic[doc_idx].sum() - 1
                    denom_a = dup_group_len * doc_len + alpha * n_topics
                else:
                    denom_a = doc_len + alpha * n_topics

                for idx, word in enumerate(words):
                    previous_topic = topic_per_docs[doc_idx][idx]

                    matrix_doc_topic[doc_idx][previous_topic] -= 1
                    matrix_word_topic[word][previous_topic] -= 1
                    matrix_master_topic[doc_idx][previous_topic] -= 1

                    denom_b = matrix_word_topic.sum(axis=0) + vocab_size * beta

                    if len(dup_group_id) == 0:
                        # Case 1: the bug has no duplicate
                        # Calculate the probability of each topic in that position: p(z_i=k|word, other_topics)
                        # Left part is the probability of the topic appears in the document doc_idx and the right part
                        # is the probability of word under a topic j
                        p_z = (matrix_doc_topic[doc_idx] + alpha) / denom_a * \
                              (matrix_word_topic[word] + beta) / denom_b
                    else:
                        # Case 2: the bug belongs to a duplicate group
                        p_z = (matrix_doc_topic[doc_idx] * matrix_master_topic[doc_idx] + alpha) / denom_a * \
                              (matrix_word_topic[word] + beta) / denom_b

                    # Sample new topic giving p_z
                    v = np.random.random() * p_z.sum()
                    c = 0

                    for new_topic, p in enumerate(p_z):
                        c += p

                        if c > v:
                            break

                    if c < v:
                        raise Exception("topic was not correctly sampled")

                    # Update the topic and the matrices
                    topic_per_docs[doc_idx][idx] = new_topic

                    matrix_master_topic[doc_idx][new_topic] += 1
                    matrix_doc_topic[doc_idx][new_topic] += 1
                    matrix_word_topic[word][new_topic] += 1

                if doc_idx > 15:
                    break

            new_theta = self.compute_theta(matrix_doc_topic)
            self.diff = np.absolute(new_theta - self.theta).sum()
            self.theta = new_theta
            logger.info("it={}. Difference between estimated topic distributions: {}".format(it, self.diff))

        self.phi = self.compute_phi(matrix_word_topic)

    def estimate_topic_proportion(self, new_docs):
        diff = math.inf
        e = 0.003

        alpha = self.alpha
        beta = self.beta
        n_topics = self.n_topics
        matrix_word_topic = self.matrix_word_topic
        vocab_size = self.vocab_size

        matrix_doc_topic = np.zeros((len(new_docs), self.n_topics))
        topic_per_docs = []

        for doc_idx, (bug_id, words) in enumerate(new_docs):
            doc_idx = doc_idx
            topics = np.random.randint(self.n_topics, size=len(words))

            for topic_idx in topics:
                matrix_doc_topic[doc_idx][topic_idx] += 1

            topic_per_docs.append(topics)

        topic_idxs = np.arange(n_topics)
        sum_topics_words = self.matrix_word_topic.sum(axis=0) - 1

        logger.info("Estimating topic proportion of {}. Threshold: {}".format([nd[0] for nd in new_docs], e))
        for _ in range(15):
        # while diff > e:
            theta = self.compute_theta(matrix_doc_topic)

            for doc_idx, (bug_id, words) in enumerate(new_docs):
                # doc_len = len(words) - 1
                doc_len = len(words)
                denom_a = doc_len + alpha * n_topics

                for idx, word in enumerate(words):
                    previous_topic = topic_per_docs[doc_idx][idx]

                    matrix_doc_topic[doc_idx][previous_topic] -= 1
                    denom_b = sum_topics_words + vocab_size * beta

                    # Case 1: the bug has no duplicate
                    # Calculate the probability of each topic in that position: p(z_i=k|word, other_topics)
                    # Left part is the probability of the topic appears in the document doc_idx and the right part
                    # is the probability of word under a topic j
                    p_z = (matrix_doc_topic[doc_idx] + alpha) / denom_a * \
                          (matrix_word_topic[word] + beta) / denom_b

                    # Sample new topic giving p_z
                    v = np.random.random() * p_z.sum()
                    c = 0

                    for new_topic, p in enumerate(p_z):
                        c += p

                        if c > v:
                            break

                    if c < v:
                        raise Exception("topic was not correctly sampled")

                    # Update the topic and the matrices
                    topic_per_docs[doc_idx][idx] = new_topic

                    matrix_doc_topic[doc_idx][new_topic] += 1

            new_theta = self.compute_theta(matrix_doc_topic)
            diff = np.absolute(new_theta - theta).mean()
            theta = new_theta
            logger.info("Mean difference between estimated topic distributions: {}".format(diff))

        for doc_idx, (bug_id, words) in enumerate(new_docs):
            self.idx_bug[bug_id] = theta[doc_idx]

        return theta

    def compare_bugs(self, new_report_id, report_ids, new_report_theta=None):
        if new_report_theta is not None:
            p = new_report_theta
        else:
            p = self.idx_bug[new_report_id]

            if isinstance(p, int):
                p = self.matrix_doc_topic[p]

        res = []
        for report_id in report_ids:
            c = self.idx_bug[report_id]

            if isinstance(c, int):
                c = self.theta[c]

            res.append(1 - jensenshannon(p, c))

        return res


class DBTM(object):

    def __init__(self, rep, tmodel):
        self.rep = rep
        self.tmodel = tmodel

    @staticmethod
    def _generate_cand_list(test_id, masterSetById, database, list_size=10000):
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

    def tune_alpha(self, query_idxs, masterSetById, database, n_candidates, dbrd_input_by_id):
        similarities = []
        queries = []

        logger = logging.getLogger()

        logger.info("Tuning alpha")

        for idx, query_id in enumerate(query_idxs):
            candidates = self._generate_cand_list(query_id, masterSetById, database, n_candidates)
            query = database.getBug(query_id)
            query_id = query['bug_id']
            dup_id = query['dup_id']

            topic_dist_reports = self.tmodel.estimate_topic_proportion(self.tmodel.transform([query]))

            query_input = dbrd_input_by_id[query_id]
            rep_sim_list = [self.rep.similarity(query_input, dbrd_input_by_id[cand]) for cand in candidates]
            tmodel_sim_list = self.tmodel.compare_bugs(query_id, candidates, topic_dist_reports[-1])

            similarities.append([(cand, rep_sim, tmodel_sim) for (cand, rep_sim, tmodel_sim) in
                                 zip(candidates, rep_sim_list, tmodel_sim_list)])

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

    def similarity(self, bug_id, bug_rep_input, candidate_id, cand_rep_input):
        #rep_sim = self.rep.similarity(bug_rep_input, cand_rep_input)
        tmodel_sim = self.tmodel.compare_bugs(bug_id, (candidate_id,))[-1]

        return tmodel_sim


class DBTMScorer(object):
    def __init__(self, dbtm, dbrd_input_by_id):
        self.dbtm = dbtm
        self.dbrd_input_by_id = dbrd_input_by_id

    def pregenerateBugEmbedding(self, allBugIds):
        pass

    def score(self, anchorBugId, bugIds):
        query_rep_inp = None
        similarities = []
        for cand_id in bugIds:
            cand_rep_inp = None

            similarities.append(self.dbtm.similarity(anchorBugId, query_rep_inp, cand_id, cand_rep_inp))

        return similarities

    def reset(self):
        pass

    def free(self):
        pass


class Run(object):

    def log_scalar(self, label, rate, step):
        pass


def train_dbtm(bugReportDatabase, base_filename, masterSetById, test, max_bug_id, rep_reports=None, tmodel_data=None):
#    input_file = path.join(args.dir, "{}_{}.txt".format(base_filename, "drbd_input"))

    # Generate file to train REP
#    if rep_reports:
#        dbrd_input, max_token_id = rep_reports
#    else:
#        dbrd_input, max_token_id = generate_input_vec(bugReportDatabase, max_bug_id)
#    logger.info(input_file)
#    dbrd_in.generate_file(dbrd_input, input_file, set(test))

    # Count the number of duplicate reports in input_file
#    n_dup_reports = 0
#    regex = re.compile(r"DID=[0-9]+")
#    with codecs.open(input_file) as dbrd_file:
#        for l in dbrd_file:
#            if regex.search(l):
#                n_dup_reports += 1

    # Trainig REP
    # ./fast-dbrd -n rep_full_2001-2007_365_validation -r /home/
    # irving/fast-dbrd/ranknet-configs/full-textual-full-categorial.cfg --training-duplicates 27481 --ts  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/timestamp_file.txt --time-constraint 365 --recommend  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/dbrd_validation.txt
#    logger.info([args.exe, "-n", args.l, "-r", args.conf, "--training-duplicates", str(n_dup_reports), input_file])
#    sp = subprocess.Popen(
#        [args.exe, "-n", args.l, "-r", args.conf, "--training-duplicates", str(n_dup_reports), input_file],
#        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=args.dir)

#    out, err = sp.communicate()

#    logger.info(out)
#    logger.info(err)

    # Load REP
#    rep = read_weights(os.path.join(args.dir, "dbrd_ranknet_{}_I-1".format(args.l)))

    # REP - transform input and compute idf and mean length
#    dbrd_input = rep.fit_transform(dbrd_input, max_token_id, False)
#    dbrd_input_by_id = {}
#    for inp in dbrd_input:
#        dbrd_input_by_id[inp[SUN_REPORT_ID_INDEX]] = inp

    if tmodel_data:
        vocab, tmodel_input = tmodel_data
        tset = set(test)
        tmodel_input = [k for k in tmodel_input if k[0] not in tset]
    else:
        # Preprocessing data for T-Model
        vocab, tmodel_input = Tmodel.preprocess(bugReportDatabase.bugList, max_bug_id, set(test), masterSetById)

    # Train T-model
    # ./fast-dbrd -n rep_full_2001-2007_365_validation -r /home/irving/fast-dbrd/ranknet-configs/full-textual-full-categorial.cfg --training-duplicates 27481 --ts  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/timestamp_file.txt --time-constraint 365 --recommend  /home/irving/workspace/duplicate_bug_report/dataset/sun_2011/eclipse_2001-2007_2008/dbrd_validation.txt
    tmodel = Tmodel(args.k, args.a, args.B)
    tmodel.fit(vocab, tmodel_input, args.it)

    # Create DBTM model
    dbtm = DBTM(None, tmodel)

    return dbtm, None


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
    parser.add_argument("-a", "--alpha", dest='a', type=float, help="")
    parser.add_argument("-B", "--beta", dest='B', type=float, help="")

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
    rep_reports = None

    if args.dt is not None:
        bugReportDatabase = BugReportDatabase.fromJson(args.db)
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

            splits= [(sorted(duplicate_reports[sample_size:], key=lambda bug_id: int(bug_id)), data_alpha)]
        elif args.fold == 0:
            logger.info("Leave one out")
            loo = LeaveOneOut()

            splits = loo.split(duplicate_reports)
        else:
            logger.info("K-folds: {}".format(args.fold))
            kf = KFold(n_splits=args.fold)

            splits = kf.split(duplicate_reports)

        base_filename = path.splitext(path.split(args.dt)[1])[0] + "_" + args.l
        max_bug_id = max(map(lambda bug_id: int(bug_id), trainingDataset.bugIds))
        masterSetById = bugReportDatabase.getMasterSetById(trainingDataset.bugIds)
        map_by_alpha = []
        n_queries = 0
        # Preprocess the reports before writing REP input file
#        rep_reports = generate_input_vec(bugReportDatabase, max_bug_id)

        # Preprocessing data for T-Model
        tmodel_data = Tmodel.preprocess(bugReportDatabase.bugList, max_bug_id, set(), masterSetById)

        for train_idx, test_idx in splits:
            break
            test = [duplicate_reports[idx] for idx in test_idx]
            dbtm, dbrd_input_by_id = train_dbtm(bugReportDatabase, base_filename, masterSetById, test, max_bug_id, rep_reports, tmodel_data)

            # Tune ensemble parameters
            map_results = dbtm.tune_alpha(test, masterSetById, bugReportDatabase, args.nc, dbrd_input_by_id)
            n_queries += len(test)


            if len(map_by_alpha) > 0:
                for idx in range(len(map_results)):
                    map_by_alpha[idx][1] += map_results[idx][1]
            else:
                map_by_alpha = map_results

#        map_by_alpha = sorted(map_by_alpha, key=lambda k: k[1], reverse=True)

#        a1 = map_by_alpha[0][0]
#        a2 = 1 - a1

#        logger.info("Best MAP: {}".format(map_by_alpha[:10]))
#        logger.info("Worst MAP: {}".format(map_by_alpha[-10:]))
#        logger.info("a1={} and a2={} achieved the best MAP({}). Number of queries={}".format(a1, a2, map_by_alpha[1], n_queries))

        dbtm, dbrd_input_by_id = train_dbtm(bugReportDatabase, base_filename, masterSetById, [], max_bug_id,  rep_reports, tmodel_data)

        dbtm.a1 = 0
        dbtm.a2 = 1

        # Save Model
        pickle.dump(dbtm, open(args.m, "wb"))

    if args.test is not None:
        bugReportDatabase = BugReportDatabase.fromJson(args.db)

        dbtm = pickle.load(open(args.m, "rb"))

        testDataset = BugDataset(args.test)
        rankingClass = SunRanking(bugReportDatabase, testDataset, args.window)
        group_by_master = True

        # Generate REP inputs
        max_bug_id = max(map(lambda bug_id: int(bug_id), testDataset.bugIds))
        #dbrd_input, max_token_id = generate_input_vec(bugReportDatabase, max_bug_id)
        #dbtm.rep.fit_transform(dbrd_input, max_token_id, True)

 #       dbrd_input_by_id = {}

  #      for inp in dbrd_input:
   #         dbrd_input_by_id[inp[SUN_REPORT_ID_INDEX]] = inp

        # Generate Tmodel inputs
        reports = dbtm.tmodel.transform(
            filter(lambda bug: bug['bug_id'] not in dbtm.tmodel.idx_bug and int(bug['bug_id']) <= max_bug_id,
                   bugReportDatabase.bugList))

        # Estimate the topic proportion of each new report
        dbtm.tmodel.estimate_topic_proportion(reports)

        # Evaluate method
        rankingScorer = DBTMScorer(dbtm, dbrd_input_by_id)

        logRankingResult(Run(), logger, rankingClass, rankingScorer, bugReportDatabase,
                         args.result_file, 0, None, group_by_master)

