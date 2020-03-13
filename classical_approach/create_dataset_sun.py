import argparse
import codecs
import logging
import os
import ujson
from datetime import datetime

import pymongo

from util.data_util import readDateFromBug, readDateFromBugWithoutTimezone


def saveFile(path, info, reports, duplicate_reports):
    f = open(path, 'w')

    f.write(info)
    f.write('\n')

    for report in reports:
        f.write('%s ' % (report["bug_id"]))

    f.write('\n')

    for dup_report in duplicate_reports:
        f.write('%s ' % (dup_report["bug_id"]))

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--database', required=True, help="dataset name")
    parser.add_argument('--collection', required=True, help="collection name")
    parser.add_argument('--start_date', required=True,
                        help="Date the start the test dataset and end the training dataset. Date in the format YYYY/mm/dd (e.g. 2018/02/10).")
    parser.add_argument('--bug_data', required=True, help="File that contains the bug report contents.")
    parser.add_argument('--test', required=True, help="test path file")
    parser.add_argument('--training', required=True, help="training path file")
    parser.add_argument('--keep_master', action="store_true", help="If this option is enable, "
                                                                   "so the original master of a set is not changed. "
                                                                   "Otherwise, we consider the first bug to be reported of a set as the master.")
    parser.add_argument('--no_tree', action="store_true", help="If this option is enable, so the script checks "
                                                               "if the masters belong to only one master set which means"
                                                               "that all masters have dup_id field equal to none")
    parser.add_argument('--train_dup', default=200, type=int)

    parser.add_argument('--end_date', required=True)
    parser.add_argument('--without_timezone', action="store_true")
    parser.add_argument('--rm_empty_report', action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    logger.addHandler(logging.StreamHandler())

    fileHandler = logging.FileHandler(os.path.join(os.path.dirname(args.bug_data), "create_dataset.log"))
    logger.addHandler(fileHandler)

    logger.info(args)

    if not args.keep_master:
        logger.warning(
            "Warning: you disabled the keep_master parameter. Be sure that all bugs belong to only 1 master set.")

    # Connect to mongo dataset
    client = pymongo.MongoClient()
    database = client[args.database]
    col = database[args.collection]

    # Read all database
    logger.info('Reading database')
    bugAndDateTuples = []
    bugsByMasterId = {}
    bugById = {}

    # Info
    info = "Database: %s; " % args.database
    info += "Order by Date; "
    info += 'Kept the original master report; ' if args.keep_master else 'Master report is the newest one'
    info += "We keep nested master reports; " if args.no_tree else "We didn't merge nested master reports"

    if args.without_timezone:
        start_date = datetime.strptime(args.start_date, '%Y/%m/%d')
        dateThreshold = datetime.strptime(args.end_date + " 23:59:59",
                                          '%Y/%m/%d %H:%M:%S') if args.end_date else None
    else:
        start_date = datetime.strptime(args.start_date + " +0000", '%Y/%m/%d %z')
        dateThreshold = datetime.strptime(args.end_date + " 23:59:59 +0000",
                                          '%Y/%m/%d %H:%M:%S %z') if args.end_date else None
    nm_report_removed = 0
    nm_empty_reports = 0

    readDateFunc = readDateFromBugWithoutTimezone if args.without_timezone else readDateFromBug

    for b in col.find({}):
        creationDate = readDateFunc(b)

        if  creationDate < start_date or creationDate > dateThreshold:
            nm_report_removed += 1
            continue

        if args.rm_empty_report and (len(b['description'].strip()) == 0 or len(b['short_desc'].strip()) == 0):
            nm_empty_reports += 1
            continue

        bugAndDateTuples.append((creationDate, b))

        bugId = b['bug_id']
        dupId = b['dup_id']
        bugById[bugId] = b

        # Put all duplicate bugs of a same set in a list
        if len(dupId) != 0:
            bugList = bugsByMasterId.get(dupId, [])

            if len(bugList) == 0:
                bugsByMasterId[dupId] = bugList

            bugList.append(b)

    if dateThreshold is not None:
        logger.warning(
            "Number of reports that were created after {}: {}".format(args.end_date, nm_report_removed))

    if args.rm_empty_report:
        logger.warning(
            "Number of reports that have summary or description empty: {}".format(nm_empty_reports))

    # Check if the master report exists in the database. If not, choose the oldest one to be the master
    for masterId, masterSet in list(bugsByMasterId.items()):
        if masterId not in bugById:
            oldestBug = masterSet[0]
            creationDateOldest = readDateFunc(masterSet[0])

            # Find first bug to be reported.
            for b in masterSet:
                creationDate = readDateFunc(b)
                # The first bugs have the same date and time of your master. We check if the id is smaller than the master
                if creationDate < creationDateOldest or (
                        creationDate == creationDateOldest and b['bug_id'] < oldestBug['bug_id']):
                    oldestBug = b
                    creationDateOldest = creationDate

            logging.warning("We didnt find the master report {} that has {} duplicate. New master {}. ".format(
                masterId, len(masterSet), oldestBug['bug_id']))

            # If first bug to reported is not the current master, so we consider it as the master
            oldestBug['dup_id'] = []
            masterSet.remove(oldestBug)

            for b in masterSet:
                # Update dup_id field to the new master report
                b['dup_id'] = oldestBug['bug_id']

            # update dup_id field of the old masterBug
            if len(masterSet) > 0:
                l = bugsByMasterId.get(oldestBug['bug_id'], [])

                if len(l) > 0:
                    logging.warning("We had to merge two master sets")

                    repeatedIds = set()
                    newMasterSet = []

                    for b in l + masterSet:
                        if b['bug_id'] not in repeatedIds:
                            newMasterSet.append(b)
                            repeatedIds.add(b['bug_id'])
                else:
                    newMasterSet = masterSet

                bugsByMasterId[oldestBug['bug_id']] = newMasterSet

            del bugsByMasterId[masterId]

    masterIdByBug = {}

    for id, bug in bugById.items():
        if len(bug['dup_id']) != 0:
            masterIdByBug[id] = bug['dup_id']

    # Remove cycles
    for id, bug in bugById.items():
        if len(bug['dup_id']) != 0:
            path = set([id, bug['dup_id']])
            nextId = masterIdByBug.get(bug['dup_id'])

            while nextId != None:
                if nextId in path:
                    # remove the next id from the master set from the next bug
                    masterSetFromNextMaster = bugsByMasterId[bugById[nextId]['dup_id']]
                    for i in range(len(masterSetFromNextMaster)):
                        if masterSetFromNextMaster[i]['bug_id'] == nextId:
                            masterSetFromNextMaster.pop(i)
                            break

                    if len(masterSetFromNextMaster) == 0:
                        del bugsByMasterId[bugById[nextId]['dup_id']]

                    bugById[nextId]['dup_id'] = []
                    del masterIdByBug[nextId]

                    logger.debug("Removing cycle {} in {}".format(path, nextId))
                    break

                path.add(nextId)
                nextId = masterIdByBug.get(nextId)

    logger.info("Cycles were removed")

    if args.no_tree:
        # Guarantee that a bug report belongs to 1 master set
        visited = set()

        nm_merged_master = 0

        for masterId, master in list(bugsByMasterId.items()):
            if masterId in visited:
                continue

            nextMasterId = masterIdByBug.get(masterId, None)
            oldMasters = []
            lastMasterId = masterId

            while nextMasterId != None:
                oldMasters.append(lastMasterId)
                lastMasterId = nextMasterId
                nm_merged_master += 1
                nextMasterId = masterIdByBug.get(nextMasterId, None)

            if len(oldMasters) != 0:
                newMasterId = lastMasterId
                newMasterSet = bugsByMasterId[newMasterId]

                for oldMasterId in oldMasters:
                    oldMaster = bugById[oldMasterId]

                    for b in bugsByMasterId[oldMasterId]:
                        if b['bug_id'] == newMasterId:
                            raise Exception()

                        b['dup_id'] = newMasterId
                        masterIdByBug[b['bug_id']] = newMasterId
                        newMasterSet.append(b)

                    visited.add(oldMaster['bug_id'])

                    del bugsByMasterId[oldMasterId]

                logger.debug("Merging {} with {}".format(oldMasters, newMasterId))

        logger.info("Number of merged master sets: {}".format(nm_merged_master))
    # Set the variables masters and masterSetIdByBug
    masterSetIdByBug = {}
    masters = set()

    for dupId, bugList in bugsByMasterId.items():
        masterBug = bugById.get(dupId)

        if not args.keep_master:
            """
            We consider the master as the bug report with the smallest creation date of the master set.
            Thus, we guarantee that master is always in the recommendation list.
            """
            oldestBug = dupId
            creationDateOldest = readDateFunc(masterBug)

            # Find first bug to be reported.
            for b in bugList:
                creationDate = readDateFunc(b)
                # The first bugs have the same date and time of your master. We check if the id is smaller than the master
                if creationDate < creationDateOldest or (
                        creationDate == creationDateOldest and int(b['bug_id']) < int(oldestBug)):
                    oldestBug = b['bug_id']
                    creationDateOldest = creationDate

            # The report id can be bigger than a report X even though X is newer than first report. Exchange the id
            masterSet = bugList + [masterBug]
            exchanged = False
            for b in masterSet:
                if int(b['bug_id']) < int(oldestBug):
                    ids = [ b['bug_id'] for b in masterSet]
                    logger.warning("Master {} has id bigger than the one of the duplicate reports: {}. Exchange the ids".format(oldestBug, ids))

                    min_id = min(ids)

                    min_bug = bugById[min_id]
                    min_bug['original_bug_id'] = min_id
                    min_bug['bug_id'] = oldestBug

                    master_bug = bugById[oldestBug]
                    master_bug['original_bug_id'] = oldestBug
                    master_bug['bug_id'] = min_id

                    oldestBug = str(min_id)
                    exchanged = True

                    break

            # If first bug to reported is not the current master, so we consider it as the master
            if oldestBug != dupId or exchanged:
                for b in bugList:
                    # Update dup_id field to the new master report
                    b['dup_id'] = [] if b['bug_id'] == oldestBug else oldestBug

                # update dup_id field of the old masterBug
                masterBug['dup_id'] = oldestBug



            dupId = oldestBug




        # Add the master report to the master set
        bugList.append(masterBug)

        for b in bugList:
            bugId = b['bug_id']
            if bugId == dupId:
                # Insert the master report to the set of master reports
                masters.add(bugId)

            # Store the dup_id of the reports in the master set (inclusive the master report)
            masterSetIdByBug[bugId] = dupId

    # Delete not updated structure
    del bugsByMasterId

    ###################################################################################
    sortedBugs = sorted(bugAndDateTuples, key=lambda tup: (tup[0], int(tup[1]['bug_id'])))

    # The index where the test begins
    testBeginIdx = None

    # List of report and duplicate report in the training
    trainingReports = []
    trainingDuplicateReports = []

    # List of report and duplicate report in the test
    testReports = []
    testDuplicateReports = []

    # Check if a report from a master set was already retrieved
    masterAlreadySeen = set()
    logger.info(
        'Split test and tranining using {} from the period {} - {}'.format(args.train_dup, start_date, dateThreshold))

    trainingReports = []
    trainingDuplicateReports = []
    nm_duplicate = 0

    if not args.without_timezone:
        logger.warning("Be careful with the timezone {} ".format(start_date.strftime(
            "%m/%d/%Y, %H:%M:%S %z")))

    for idx, (date, bug) in enumerate(sortedBugs):
        masterId = masterSetIdByBug.get(bug['bug_id'])

        if nm_duplicate < args.train_dup:
            trainingReports.append(bug)
            l = trainingDuplicateReports
        else:
            testReports.append(bug)
            l = testDuplicateReports

        if masterId is None:
            # It is not duplicate or it is not master report that contains duplicate reports
            continue

        # A report is only considered duplicate when another report from the same master set has already been retrieved before.
        if masterId in masterAlreadySeen:
            l.append(bug)
            nm_duplicate += 1
        else:
            masterAlreadySeen.add(masterId)

    logger.info(
        'Total:\t%d duplicate bugs\t%d of bug reports' % (
            len(trainingDuplicateReports) + len(testDuplicateReports), len(sortedBugs)))
    logger.info(
        "Training:\t%d duplicate bugs\t%d of bug reports" % (len(trainingDuplicateReports), len(trainingReports)))
    logger.info("Test:\t%d duplicate bugs\t%d of bug reports" % (len(testDuplicateReports), len(testReports)))

    saveFile(args.training, info, trainingReports, trainingDuplicateReports)
    saveFile(args.test, info, testReports, testDuplicateReports)

    logger.info('Saving json file')

    jsonFile = codecs.open(args.bug_data, 'w', encoding="utf-8")

    for date, bug in sortedBugs:
        # Saving bug in txt file
        del bug['_id']

        if isinstance(bug['description'], list):
            bug['description'] = ""
            logger.info("{} has a list as description".format(bug['bug_id']))

        if isinstance(bug['short_desc'], list):
            bug['short_desc'] = ""
            logger.info("{} has a list as summary".format(bug['bug_id']))

        jsonFile.write(ujson.dumps(bug))

        jsonFile.write('\n')

    logger.info("Finished!!!")
