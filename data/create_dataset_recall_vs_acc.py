"""
This script creates the dataset used to perform the experiment that relates recall rate and accuracy.

"""
import argparse
import json
import random

import ujson

from data.create_dataset_deshmukh import getMasterSetAndDuplicateByMaster, createKey, createNonDuplicateBugGenerator, \
    generateNonDuplicatePairs, hasDuplicatePairs


def loadPairs(dataset, pairsDict={}, ignoreNonDuplicates=False, removeDuplicateExamples=True):
    duplicatePairs = []
    nonDuplicatePairs = []
    nbRemovedDupEx = 0

    for l in open(dataset, 'r'):
        bug1, bug2, label = l.strip().split(',')
        label = int(label)

        if label == 1:
            if removeDuplicateExamples and createKey(bug1, bug2) in pairsDict:
                nbRemovedDupEx += 1
                continue

            duplicatePairs.append((bug1, bug2, 1))
            pairsDict[createKey(bug1, bug2)] = True
        elif label == -1:
            if not ignoreNonDuplicates:
                nonDuplicatePairs.append((bug1, bug2, -1))
                pairsDict[createKey(bug1, bug2)] = True
        else:
            raise Exception(
                "Label has a value different of -1 or 1. You might have used the file with triplet instead of pairs")

    print('Amount of examples duplicated: %d' % nbRemovedDupEx)

    return duplicatePairs, nonDuplicatePairs, pairsDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--proportions', nargs='+', help="proportion of nonduplicate pairs")
    parser.add_argument('--training', help="dataset used to training the models")
    parser.add_argument('--validation', help="dataset used to validate the models")
    parser.add_argument('--database', help="dataset name")
    parser.add_argument('--collection', help="dataset name")
    parser.add_argument('--input', help="final file")
    parser.add_argument('--output', required=True, help="final file")

    args = parser.parse_args()
    print(args)

    if args.input:
        raise NotImplementedError("")

    js = {
        'training': args.training,
        'validation': args.validation,
        'database': args.database,
        'collection': args.collection
    }

    masterSet, duplicateByMasterId = getMasterSetAndDuplicateByMaster(args.database, args.collection)
    trainingDuplicate, trainingNonDuplicate, allPairDict = loadPairs(args.training)
    validationDuplicate, _, allPairDict = loadPairs(args.validation, allPairDict, True)

    proportions = sorted([int(p) for p in args.proportions])
    validationNonDuplicate = generateNonDuplicatePairs(validationDuplicate, masterSet, proportions[-1], allPairDict)

    output = open(args.output, 'w')
    trainingPairs = trainingDuplicate + trainingNonDuplicate
    validationPairs = validationDuplicate + validationNonDuplicate

    for pair in validationPairs:
        output.write('%s,%s,%s\n' % (pair[0], pair[1], pair[2]))

    output.write('\n')

    seed = 987276263637
    startNonDuplicate = len(validationDuplicate)
    biggestProportion = proportions[-1]

    if hasDuplicatePairs(trainingPairs, validationPairs, [], True):
        raise Exception('There are duplicate pairs in the dataset!!')

    data = []

    for p in proportions[:-1]:
        random.seed(seed + p)

        po = {'k': p}
        indexes = [i for i in range(startNonDuplicate)]
        probabilityGetPair = p / float(biggestProportion)

        nNonDuplicatePairs = 0

        for i in range(startNonDuplicate, startNonDuplicate + len(validationNonDuplicate)):
            a = random.random()
            if probabilityGetPair > a:
                indexes.append(i)
                nNonDuplicatePairs += 1

        po['indexes'] = indexes
        data.append(po)

        print("Proportion %s\tAmount of Duplicate: %d\tAmount of Nonduplicate: %d\tTotal: %d" % (
            p, startNonDuplicate, nNonDuplicatePairs, len(indexes)))

    po = {'k': biggestProportion, 'indexes': list(range(len(validationDuplicate) + len(validationNonDuplicate)))}
    data.append(po)

    js['validations'] = data

    output.write(ujson.dumps(js))

    print("Proportion %s\tAmount of Duplicate: %d\tAmount of Nonduplicate: %d\tTotal: %d" % (
        biggestProportion, len(validationDuplicate), len(validationNonDuplicate),
        len(validationDuplicate) + len(validationNonDuplicate)))
