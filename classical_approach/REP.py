import logging
import math

from classical_approach.bm25f import SUN_REPORT_VERSION, SUN_REPORT_COMPONENT, SUN_REPORT_SUB_COMPONENT, \
    SUN_REPORT_TYPE, SUN_REPORT_PRIORITY


class REP(object):

    def __init__(self, bm25f, w_component, w_sub_component, w_type, w_priority, w_version):
        self.w_component = w_component
        self.w_sub_component =w_sub_component
        self.w_type = w_type
        self.w_priority = w_priority
        self.w_version = w_version

        logging.getLogger().info("REP weights: {}".format(self.__dict__))

        self.bm25f = bm25f

    def fit_transform(self, reports, max_token_id, overwrite_reports=True):
        return self.bm25f.fit_transform(reports, max_token_id, overwrite_reports)

    def similarity(self, query, candidate):
        score = self.bm25f.similarity(query,candidate)

        score += self.w_component * int(query[SUN_REPORT_COMPONENT] == candidate[SUN_REPORT_COMPONENT])
        score += self.w_sub_component * int(query[SUN_REPORT_SUB_COMPONENT] == candidate[SUN_REPORT_SUB_COMPONENT])
        score += self.w_type * int(query[SUN_REPORT_TYPE] == candidate[SUN_REPORT_TYPE])
        score += self.w_version * (1 / (1 + math.fabs(query[SUN_REPORT_VERSION] - candidate[SUN_REPORT_VERSION])))
        score += self.w_priority * (1 / (1 + math.fabs(query[SUN_REPORT_PRIORITY] - candidate[SUN_REPORT_PRIORITY])))

        return score


