import logging
from math import log
from time import time
import numpy as np

SUN_REPORT_ID_INDEX = 0
SUN_REPORT_DID_INDEX = 1
SUN_REPORT_UNIGRAM_TOKEN = 2
SUN_REPORT_UNIGRAM_SUM = 3
SUN_REPORT_UNIGRAM_SUM_LEN = 4
SUN_REPORT_UNIGRAM_DESC = 5
SUN_REPORT_UNIGRAM_DESC_LEN = 6
SUN_REPORT_BIGRAM_TOKEN = 7
SUN_REPORT_BIGRAM_SUM = 8
SUN_REPORT_BIGRAM_SUM_LEN = 9
SUN_REPORT_BIGRAM_DESC = 10
SUN_REPORT_BIGRAM_DESC_LEN = 11
SUN_REPORT_VERSION = 12
SUN_REPORT_COMPONENT = 13
SUN_REPORT_SUB_COMPONENT = 14
SUN_REPORT_TYPE = 15
SUN_REPORT_PRIORITY = 16


def calculate_length(tokens):
    len = 0

    for _, freq in tokens:
        len += freq

    return len


def score_bm25f(query_tk, query_sum, query_desc, cand_tk, cand_sum, cand_sum_len,
                cand_desc, cand_desc_len, k1, k3, wf_list, bf_list, avg_lengths, IDF):
    query_idx = 0
    cand_idx = 0
    cand_len = (cand_sum_len, cand_desc_len)
    bm25f_score = 0

    n_query = len(query_tk)
    n_cand = len(cand_tk)

    while query_idx < n_query and cand_idx < n_cand:
        query_token_id = query_tk[query_idx]
        cand_token_id = cand_tk[cand_idx]

        if query_token_id < cand_token_id:
            query_idx += 1
        elif query_token_id > cand_token_id:
            cand_idx += 1
        else:
            # Compute term and query weight BM25F
            term_weight = 0
            query_weight = 0

            query_tf_values = (float(query_sum[query_idx]), float(query_desc[query_idx]))
            cand_tf_values = (float(cand_sum[cand_idx]), float(cand_desc[cand_idx]))

            for qtf, ctf, wf, bf, length, avg_len in zip(query_tf_values, cand_tf_values, wf_list, bf_list, cand_len,
                                                         avg_lengths):
                if ctf != 0:
                    term_weight += (wf * ctf) / ((1 - bf) + bf * length / avg_len)

                if qtf != 0:
                    query_weight += wf * qtf

            # Compute query weight BM25F
            query_part = ((k3 + 1) * query_weight) / (k3 + query_weight)
            term__part = term_weight / (k1 + term_weight)
            bm25f_score += IDF[query_token_id] * term__part * query_part

            cand_idx += 1
            query_idx += 1

    return bm25f_score


class BM25F_EXT:

    def __init__(self, total_weight_unigram, total_weight_bigram, k1_unigram, k3_unigram, w_unigram_sum, w_unigram_desc,
                 bf_unigram_sum, bf_unigram_desc, k1_bigram, k3_bigram, w_bigram_sum, w_bigram_desc, bf_bigram_sum,
                 bf_bigram_desc):
        self.k1_unigram = k1_unigram
        self.k3_unigram = k3_unigram
        self.w_unigram = (w_unigram_sum, w_unigram_desc)
        self.bf_unigram = (bf_unigram_sum, bf_unigram_desc)
        self.total_weight_unigram = total_weight_unigram

        self.k1_bigram = k1_bigram
        self.k3_bigram = k3_bigram
        self.w_bigram = (w_bigram_sum, w_bigram_desc)
        self.bf_bigram = (bf_bigram_sum, bf_bigram_desc)
        self.total_weight_bigram = total_weight_bigram

        logging.getLogger().info("BM25F weights: {}".format(self.__dict__))

        self.IDF = {}
        self.unigram_average = [0., 0.]
        self.bigram_average = [0., 0.]

    @staticmethod
    def aggregate_field_tf(summary, description):
        sum_idx = 0
        desc_idx = 0
        sum_len = len(summary)
        desc_len = len(description)

        union_tk = []
        union_sum = []
        union_desc = []

        while sum_idx < sum_len and desc_idx < desc_len:
            sum_token_id, sum_token_tf = summary[sum_idx]
            desc_token_id, desc_token_tf = description[desc_idx]

            if sum_token_id > desc_token_id:
                union_tk.append(desc_token_id)
                union_sum.append(0)
                union_desc.append(desc_token_tf)

                desc_idx += 1
            elif desc_token_id > sum_token_id:
                union_tk.append(sum_token_id)
                union_sum.append(sum_token_tf)
                union_desc.append(0)

                sum_idx += 1
            else:
                union_tk.append(sum_token_id)
                union_sum.append(sum_token_tf)
                union_desc.append(desc_token_tf)

                desc_idx += 1
                sum_idx += 1

        while sum_idx < sum_len:
            sum_token_id, sum_token_tf = summary[sum_idx]
            union_tk.append(sum_token_id)
            union_sum.append(sum_token_tf)
            union_desc.append(0)
            sum_idx += 1

        while desc_idx < desc_len:
            desc_token_id, desc_token_tf = description[desc_idx]

            union_tk.append(desc_token_id)
            union_sum.append(0)
            union_desc.append(desc_token_tf)

            desc_idx += 1

        return np.asarray(union_tk, dtype=np.uint32), np.asarray(union_sum, dtype=np.uint8), sum(union_sum), \
               np.asarray(union_desc, dtype=np.uint8), sum(union_desc)

    def fit_transform(self, reports, max_token_id, replace_reports=True):
        n_reports = 0
        self.IDF = [0.0] * max_token_id

        new_reports = reports if replace_reports else [None] * len(reports)

        for idx, report in enumerate(reports):
            for token_id, tf in report['A-U']:
                self.IDF[token_id] += 1.0

            new = [report['id'], report['DID']]

            import sys

            sys.getsizeof(report['id'])

            self.unigram_average[0] += calculate_length(report['S-U'])
            self.unigram_average[1] += calculate_length(report['D-U'])
            new.extend(self.aggregate_field_tf(report['S-U'], report['D-U']))

            for token_id, tf in report['A-B']:
                self.IDF[token_id] += 1.0

            self.bigram_average[0] += calculate_length(report['S-B'])
            self.bigram_average[1] += calculate_length(report['D-B'])
            new.extend(self.aggregate_field_tf(report['S-B'], report['D-B']))

            new.extend(
                [report['VERSION'], report['COMPONENT'], report['SUB-COMPONENT'], report['TYPE'], report['PRIORITY']])

            new_reports[idx] = new
            n_reports += 1

        self.unigram_average[0] /= n_reports
        self.unigram_average[1] /= n_reports
        self.bigram_average[0] /= n_reports
        self.bigram_average[1] /= n_reports

        for token_id in range(len(self.IDF)):
            if self.IDF[token_id] == 0:
                continue

            self.IDF[token_id] = log(n_reports / self.IDF[token_id], 2)

        logging.getLogger().info(
            "Reports: {}, Unigram Average: {}, Bigram Average: {}".format(len(reports), self.unigram_average,
                                                                          self.bigram_average))

        return new_reports

    def similarity(self, query, candidate):
        scores = []
        query_unigram = query[SUN_REPORT_UNIGRAM_TOKEN]
        query_unigram_sum = query[SUN_REPORT_UNIGRAM_SUM]
        query_unigram_sum_len = query[SUN_REPORT_UNIGRAM_SUM_LEN]

        query_unigram_desc = query[SUN_REPORT_UNIGRAM_DESC]
        query_unigram_desc_len = query[SUN_REPORT_UNIGRAM_DESC_LEN]

        query_bigram = query[SUN_REPORT_BIGRAM_TOKEN]
        query_bigram_sum = query[SUN_REPORT_BIGRAM_SUM]
        query_bigram_sum_len = query[SUN_REPORT_BIGRAM_SUM_LEN]

        query_bigram_desc = query[SUN_REPORT_BIGRAM_DESC]
        query_bigram_desc_len = query[SUN_REPORT_BIGRAM_DESC_LEN]

        candidate_unigram = candidate[SUN_REPORT_UNIGRAM_TOKEN]
        candidate_unigram_sum = candidate[SUN_REPORT_UNIGRAM_SUM]
        candidate_unigram_sum_len = candidate[SUN_REPORT_UNIGRAM_SUM_LEN]

        candidate_unigram_desc = candidate[SUN_REPORT_UNIGRAM_DESC]
        candidate_unigram_desc_len = candidate[SUN_REPORT_UNIGRAM_DESC_LEN]

        candidate_bigram = candidate[SUN_REPORT_BIGRAM_TOKEN]

        candidate_bigram_sum = candidate[SUN_REPORT_BIGRAM_SUM]
        candidate_bigram_sum_len = candidate[SUN_REPORT_BIGRAM_SUM_LEN]

        candidate_bigram_desc = candidate[SUN_REPORT_BIGRAM_DESC]
        candidate_bigram_desc_len = candidate[SUN_REPORT_BIGRAM_DESC_LEN]

        unigram_score = score_bm25f(query_unigram, query_unigram_sum, query_unigram_desc, candidate_unigram,
                                    candidate_unigram_sum, candidate_unigram_sum_len, candidate_unigram_desc,
                                    candidate_unigram_desc_len, self.k1_unigram, self.k3_unigram, self.w_unigram,
                                    self.bf_unigram, self.unigram_average, self.IDF)
        bi_score = score_bm25f(query_bigram, query_bigram_sum, query_bigram_desc, candidate_bigram,
                               candidate_bigram_sum, candidate_bigram_sum_len, candidate_bigram_desc,
                               candidate_bigram_desc_len, self.k1_bigram, self.k3_bigram, self.w_bigram, self.bf_bigram,
                               self.bigram_average, self.IDF)
        return self.total_weight_unigram * unigram_score + self.total_weight_bigram * bi_score
