import sys

from data.bug_report_database import BugReportDatabase
from data.create_dataset_our_methodology import saveFile

"""
Convert Old dataset file to new
"""

bugDatabase = BugReportDatabase.fromJson(sys.argv[1])

oldFile = open(sys.argv[2], 'r')

start, end = map(lambda k: int(k), oldFile.readline().strip().split(' '))
duplicateIdxs = [int(idx) for idx in oldFile.readline().strip().split()]

oldFile.close()

sortedBugs = [ ('t',bug) for bug in bugDatabase.bugList]
info = "Dataset: eclipse; Order by date; Master is the oldest one; No nested master"

saveFile(sys.argv[2], info, start, end, duplicateIdxs, sortedBugs)