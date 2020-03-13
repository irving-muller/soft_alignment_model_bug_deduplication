import argparse
import codecs
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-ts', required=True, help="")
    parser.add_argument('-v', required=True, help="")
    parser.add_argument('-tsp', required=True, help="")
    parser.add_argument('-vp', required=True, help="")
    parser.add_argument('-tst', required=True, help="")
    parser.add_argument('-vt', required=True, help="")

    parser.add_argument('-t', required=True, help="")
    parser.add_argument('-tp', required=True, help="")
    parser.add_argument('-tt', required=True, help="")

    args = parser.parse_args()

    training_file = codecs.open(args.t, 'w')
    training_pairs_file = codecs.open(args.tp, 'w')
    training_triplets_file = codecs.open(args.tt, 'w')

    training_split_file = codecs.open(args.ts,'r')
    training_split_pairs_file = codecs.open(args.tsp, 'r')
    training_split_triplets_file = codecs.open(args.tst, 'r')

    validation_file = codecs.open(args.v, 'r')
    validation_pairs_file = codecs.open(args.vp, 'r')
    validation_triplets_file = codecs.open(args.vt, 'r')

    for l in training_split_pairs_file:
        training_pairs_file.write(l)

    for l in validation_pairs_file:
        training_pairs_file.write(l)

    for l in training_split_triplets_file:
        training_triplets_file.write(l)

    for l in validation_triplets_file:
        training_triplets_file.write(l)

    training_file.write(training_split_file.readline().rstrip())
    training_file.write(" ")
    training_file.write(validation_file.readline().rstrip())
    training_file.write("\n")

    training_file.write(training_split_file.readline().rstrip())
    training_file.write(" ")
    training_file.write(validation_file.readline().rstrip())
    training_file.write("\n")

    training_file.write(training_split_file.readline().rstrip())
    training_file.write(" ")
    training_file.write(validation_file.readline().rstrip())
    training_file.write("\n")