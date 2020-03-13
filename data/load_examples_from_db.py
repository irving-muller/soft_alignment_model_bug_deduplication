# Spliting the pairs in training and test dataset
import argparse
import codecs
import ujson

import pymongo

parser = argparse.ArgumentParser()
parser.add_argument('--file', required="True", help="")
parser.add_argument('--triplet', action="store_true", help="")
parser.add_argument('--dataset', required="True", help="")
parser.add_argument('--collection', required="True", help="")
parser.add_argument('--save', required="True", help="")

args = parser.parse_args()

print(args.file)

f = open(args.file, 'r')

bugIds = {}

for l in f:
    if len(l.strip()) == 0:
        break

    bug1, bug2, label = l.strip().split(',')
    bugIds[bug1] = True
    bugIds[bug2] = True

    if args.triplet:
        bugIds[label] = True

client = pymongo.MongoClient()
collection = client[args.dataset][args.collection]

cursor = collection.find({}).batch_size(20000)

f = codecs.open(args.save, 'w', encoding="utf-8")

alreadyDownloaded = {}

total = 0

for bug in cursor:
    if bugIds.get(bug['bug_id'], False) and not alreadyDownloaded.get(bug['bug_id'], False):
        # Removing id from the database
        bug.pop('_id')
        # Converting to string
        j = ujson.dumps(bug)

        alreadyDownloaded[bug['bug_id']] = True

        # Saving bug in txt file
        f.write(j)
        f.write('\n')

        total+=1

print(total)
print("File save with success!!")
