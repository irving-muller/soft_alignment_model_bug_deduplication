import argparse
import logging
import sys
import numpy as np

import ujson

import math

from data.bug_report_database import BugReportDatabase
from util.data_util import readDateFromBug

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', required=True, help="")
    parser.add_argument('--bug_dataset', required=True, help="")

    args = parser.parse_args()
    print(args)

    bugReportDatabase = BugReportDatabase.fromJson(args.bug_dataset)
    results = open(args.f, 'r')

    ks = list(range(1, 21))
    hits = [0] * len(ks)
    hitsPerYear = {}

    nMasterFirst = 0
    nms = 0
    nm = 0
    masterTotal = 0

    masterSetId = bugReportDatabase.getMasterSetById()
    nResults = 0

    for l in results:
        obj = ujson.loads(l)
        bug = bugReportDatabase.getBug(obj['id'])
        masterId = bug['dup_id'] if len(bug['dup_id']) > 0 else bug['bug_id']
        creationDate = readDateFromBug(bug)
        yearTuple = hitsPerYear.get(creationDate.year, [[0] * len(ks), 0])

        if yearTuple[1] == 0:
            hitsPerYear[creationDate.year] = yearTuple

        yearTuple[1] += 1

        smallPosition = math.inf
        masterPosition = None

        for bugId, pos, score in obj['pos']:
            if masterId == bugId:
                masterPosition = pos

            if pos < smallPosition:
                smallPosition = pos

        if masterPosition is not None:
            if masterPosition <= smallPosition:
                nMasterFirst += 1

            if len(obj['pos']) > 1:
                if masterPosition <= smallPosition:
                    nm += 1

                nms += 1

            masterTotal += 1

        for idx, k in enumerate(ks):
            if k < smallPosition:
                continue

            hits[idx] += 1
            yearTuple[0][idx] += 1

        nResults += 1

    print("##############################################")
    print("Recall Rate Results:")
    recallRate = []
    for k, hit in zip(ks, hits):
        rate = float(hit) / nResults
        recallRate.append(rate)
        print("\t\t k=%d: %.3f (%d/%d) " % (k, rate, hit, nResults))
    print('-------')
    print(recallRate)
    print("##############################################")
    print("Recall Rate Per Year:")
    recallRatePerYer = []
    for year, (yearHits, yearTotal) in sorted(hitsPerYear.items(), key=lambda t: t[0]):
        print("\t%s" % year)

        recallRate = []
        for k, hit in zip(ks, yearHits):
            rate = float(hit) / yearTotal
            recallRate.append(rate)
            print("\t\t k=%d: %.3f (%d/%d) " % (k, rate, hit, yearTotal))

        recallRatePerYer.append((year, recallRate))

    print('-------')
    print(recallRatePerYer)
    print("##############################################")
    print("Total of master bugs that your position were better than the duplicate bugs: %d of %d" % (
        nMasterFirst, masterTotal))
    print(
        "Amount of times that master set had the best position and the recommendation list had the master bug and at least one duplicate bug: %d of %d" % (
            nm, nms))
