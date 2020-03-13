"""
This script converts glove text file into a binary file. It creates two files: lexicon file and vectors file.

Example:

python transform_glove_binary.py DIR_PATH/glove_42B_300d_mozilla_2001-2009_2010_soft_clean_merged.txt
DIR_PATH/glove_42B_300d_mozilla_2001-2009_2010_soft_clean_merged.npy
DIR_PATH/glove_42B_300d_mozilla_2001-2009_2010_soft_clean_merged.lxc
-db DIR_PATH/mozilla_2001-2009_2010/mozilla_soft_clean_rm_punc_sent_tok.txt.json
-tk white_space
-filters TransformLowerCaseFilter
"""

import argparse
import codecs
import logging
from argparse import ArgumentError

import numpy
from nltk import WhitespaceTokenizer

from data.Embedding import Embedding
from data.bug_report_database import BugReportDatabase
from data.preprocessing import loadFilters, MultiLineTokenizer


def extract_tokens(text, tokenizer, filters):
    if isinstance(text, list):
        return

    tokens = tokenizer.tokenize(text)

    for token in tokens:
        for filter in filters:
            token = filter.filter(token, None)

        yield token


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('word_embedding', help="glove text file")
    parser.add_argument('output', help="file where the vector will be saved")
    parser.add_argument('lexicon', help="file where the lexicon will be saved")
    parser.add_argument('-db', help="File path that contains all reports. "
                                    "If this option is set, so tokens that not appear in any report are removed from the lexicon.")
    parser.add_argument('-filters', default=[], nargs='+', help="text filters")
    parser.add_argument('-tk', help="tokenizer")

    logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)
    lexicon, embedding = Embedding.fromFile(args.word_embedding, 'UUUKNNN', hasHeader=False,
                                            paddingSym="</s>")
    database_lexicon = set(['UUUKNNN', '</s>'])

    if args.db is not None:
        # Filter any word that is not in the dataset.
        db = BugReportDatabase.fromJson(args.db)
        filters = loadFilters(args.filters)

        if args.tk == 'default':
            logger.info("Use default tokenizer to tokenize summary information")
            tokenizer = MultiLineTokenizer()
        elif args.tk == 'white_space':
            logger.info("Use white space tokenizer to tokenize summary information")
            tokenizer = WhitespaceTokenizer()
        else:
            raise ArgumentError(
                "Tokenizer value %s is invalid. You should choose one of these: default and white_space" % args.tk)

        for report in db.bugList:
            for token in extract_tokens(report.get('short_desc', ''), tokenizer, filters):
                database_lexicon.add(token)

            for token in extract_tokens(report.get('description', ''), tokenizer, filters):
                database_lexicon.add(token)

    embeddings = []
    f = codecs.open(args.lexicon, 'w')
    n_removed_words = 0

    for word, emb in zip(lexicon.getLexiconList(), embedding.getEmbeddingMatrix()):
        if args.db is not None and word not in database_lexicon:
            n_removed_words += 1
            continue

        f.write(word)
        f.write('\n')

        embeddings.append(emb)

    f.close()
    numpy.save(args.output, numpy.asarray(embeddings))

    print('Total embeddings: {}'.format(len(lexicon.getLexiconList())))
    print('Total of removed embeddings: {}'.format(n_removed_words))
