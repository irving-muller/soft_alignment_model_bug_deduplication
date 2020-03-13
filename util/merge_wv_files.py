"""
Merge two files created by glove
"""

import argparse

parser = argparse.ArgumentParser()

# Global arguments
parser.add_argument('--main', required=True, help="Main glove file")
parser.add_argument('--aux', required=True, help="Additional word embeddings")
parser.add_argument('--output', required=True, help="")

args = parser.parse_args()

outFile = open(args.output, 'w')
mainFile = open(args.main, 'r')

wordSet = set()

for l in mainFile:
    word = l.split(' ')[0]
    wordSet.add(word)

    outFile.write(l)

outFile.write('\n')
mainFile.close()

auxFile = open(args.aux, 'r')

for idx,l in enumerate(auxFile):
    word = l.split(' ')[0]

    if word in wordSet:
        continue

    outFile.write(l)
