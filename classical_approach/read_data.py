import codecs
import re

from classical_approach.REP import REP
from classical_approach.bm25f import BM25F_EXT


def read_weights(file_path):
    file = codecs.open(file_path)

    unigram = False
    bm25f_input = {}
    REP_input = {}

    pattern = re.compile(r'\] = ([0-9.-]+)')
    var = None
    var2 = None

    for l in file:
        if '[unigram] BM25F Parameter...' in l:
            unigram = True
            continue
        elif '[bigram] BM25F Parameter...' in l:
            unigram = False
            continue
        elif 'total weight' in l:
            var = 'total_weight_unigram' if unigram else 'total_weight_bigram'
        elif '[k1,' in l:
            var = 'k1_unigram' if unigram else 'k1_bigram'
        elif '[summary weight' in l:
            var = 'w_unigram_sum' if unigram else 'w_bigram_sum'
        elif '[summary b' in l:
            var = 'bf_unigram_sum' if unigram else 'bf_bigram_sum'
        elif '[description weight' in l:
            var = 'w_unigram_desc' if unigram else 'w_bigram_desc'
        elif '[description b' in l:
            var = 'bf_unigram_desc' if unigram else 'bf_bigram_desc'
        elif '[k3,' in l:
            var = 'k3_unigram' if unigram else 'k3_bigram'
        elif '[component weight' in l:
            REP_input['w_component'] = float(pattern.search(l).group(1))
            continue
        elif '[sub component weight' in l:
            REP_input['w_sub_component'] = float(pattern.search(l).group(1))
            continue
        elif '[report type weight' in l:
            REP_input['w_type'] = float(pattern.search(l).group(1))
            continue
        elif '[priority weight' in l:
            REP_input['w_priority'] = float(pattern.search(l).group(1))
            continue
        elif '[version weight' in l:
            REP_input['w_version'] = float(pattern.search(l).group(1))
            continue
        else:
            continue

        if var:
            bm25f_input[var] = float(pattern.search(l).group(1))

    bm25f = BM25F_EXT(**bm25f_input)

    for v in REP_input.values():
        if v > 0.0:
            REP_input['bm25f'] = bm25f
            return REP(**REP_input)

    return bm25f


def read_dbrd_file(dbrd_input, max_bug_id):
    report = None
    reports = []
    categorical_fields = ['DID', 'VERSION', 'COMPONENT', 'SUB-COMPONENT', 'TYPE', 'PRIORITY']
    max_token_id = -1

    for l in codecs.open(dbrd_input):
        field_name, field_value = l.split('=', maxsplit=1)
        field_value = field_value.strip()

        if field_name == 'ID':
            if int(field_value) > max_bug_id:
                break

            if report is not None:
                reports.append(report)

            report = {"id": field_value}
        elif field_name in categorical_fields:
            if field_name != 'DID':
                field_value = int(field_value)

            report[field_name] = field_value
        else:
            tfs = []

            if len(field_value) > 0:
                for token_id_tf in field_value.split(','):
                    token_id, tf = token_id_tf.split(':')
                    token_id = int(token_id)

                    tfs.append((token_id, int(tf)))

                    if token_id > max_token_id:
                        max_token_id = token_id

            report[field_name] = tfs

    reports.append(report)

    return reports, max_token_id
