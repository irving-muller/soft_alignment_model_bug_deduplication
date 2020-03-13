import argparse
import codecs
import logging
import pickle
from _ctypes import ArgumentError
from collections import OrderedDict

import numpy
import tqdm
from nltk import WhitespaceTokenizer

from data import Embedding
from data.Embedding import generateVector
from data.bug_report_database import BugReportDatabase
from data.preprocessing import MultiLineTokenizer, loadFilters, TextFieldPreprocessor, checkDesc
from fasttext import load_model
import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('db',  help="")
    parser.add_argument('model', help="")
    parser.add_argument('lexicon', help="")
    parser.add_argument('wv', help="")
    parser.add_argument('-filters', nargs='+', help="")
    parser.add_argument('-tk', help="")

    logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)

    db = BugReportDatabase.fromJson(args.db)

    # Tokenizer
    if args.tk == 'default':
        logger.info("Use default tokenizer to tokenize summary information")
        tokenizer = MultiLineTokenizer()
    elif args.tk == 'white_space':
        logger.info("Use white space tokenizer to tokenize summary information")
        tokenizer = WhitespaceTokenizer()
    else:
        raise ArgumentError(
            "Tokenizer value %s is invalid. You should choose one of these: default and white_space" %
            args.tk)

    extractorFilters = loadFilters(args.filters)
    preprocessing = TextFieldPreprocessor(None, extractorFilters, tokenizer)

    model = load_model(args.model)
    lexicon = OrderedDict()
    wvs = []

    lexicon["UUUKNNN"] = True
    wvs.append(list(generateVector(model.get_dimension())))

    lexicon["</s>"] = True
    wvs.append([0.0] * model.get_dimension())


    for bug in tqdm.tqdm(db.bugList):
        summary = bug['short_desc'].strip()

        if len(summary) == 0:
            summary = []
        else:
            summary = preprocessing.preprocess(summary)

        description = checkDesc(bug['description'])

        if len(description) == 0:
            description = []
        else:
            description = preprocessing.preprocess(description)

        if len(summary) + len(description) == 0:
            continue

        sum_desc = summary + description

        for word in sum_desc:
            if word in lexicon:
                continue

            lexicon[word] = True
            wvs.append(list(model.get_word_vector(word)))



    wvs = numpy.asarray(wvs)
    numpy.save(args.wv, wvs)

    f = codecs.open(args.lexicon, 'w')

    for word in lexicon.keys():
        f.write(word)
        f.write('\n')

    f.close()
