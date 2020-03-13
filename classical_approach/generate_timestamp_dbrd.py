"""
Generate the timestamp file that is used by REP and BM25F_ext
Example
python generate_timestamp_dbrd
    --database DATASET_DIR/sun_2011/mozilla_2001-2009_2010/mozilla_initial.json
    --ts DATASET_DIR/sun_2011/mozilla_2001-2009_2010/timestamp_file.txt
"""
import argparse
import codecs
import logging

from data.bug_report_database import BugReportDatabase
from util.data_util import readDateFromBug

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--database', required=True, help="")
    parser.add_argument('--ts', required=True, help="")

    logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)

    database = BugReportDatabase.fromJson(args.database)
    ts_file = codecs.open(args.ts, 'w')

    for bug_id, bug in sorted(map(lambda item: (int(item[0]), item[1]), database.bugById.items())):
        bug_ts = readDateFromBug(bug).timestamp()
        # Timestamp in days
        ts_file.write("{}={}\n".format(bug['bug_id'], int(bug_ts/(24*60*60))))


