import argparse
import itertools
import json
import random
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt

import pymongo
import re

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('mongoDB', help='mongo DB')
parser.add_argument('query', help='query')
parser.add_argument('-p', nargs='*', default=None, help='Parameters to print')
parser.add_argument('-m', default=None, help='Metric name to be plotted')
parser.add_argument('-s', default=None, help='Sort by metric')
parser.add_argument('-n', action='store_true', help='')
parser.add_argument('-g', default=None, help='Group metrics using a regex')

args = parser.parse_args()

client = pymongo.MongoClient()
database = client[args.mongoDB]

runCollection = database['runs']
metricCollection = database['metrics']

query = eval(args.query)

print(query)
print('Query:{}'.format(query))

result = {}

for res in runCollection.find(query).sort('_id',pymongo.ASCENDING):
    metricResults = metricCollection.find({'run_id': res['_id']})

    if metricResults.count() == 0:
        continue

    print('##########ID: %d (%s)##############' % (res['_id'], args.mongoDB))
    if args.p:
        expArgs = OrderedDict()

        if args.n:
            expArgs['exp_name'] = res['experiment']['name']

        for param in args.p:
            paramValue = res['config']

            for p in param.split('.'):
                paramValue = paramValue.get(p)

                if  paramValue is None:
                    break

            expArgs[param] = paramValue
    else:
        expArgs = res['config']

    print('""" config={} """'.format(json.dumps(expArgs)))

    metricToPlot = []

    if args.g:
        group_regex = re.compile(args.g)
        groups = {}

    for metric in metricResults:
        print('{}={}'.format(metric['name'], metric['values']))

        if metric['name'] == args.m or metric['name'] == args.s:
            metricToPlot.append(metric)
        if args.g:
           group_regex = re.compile(args.g)
           g = group_regex.search(metric['name'])
           if g:
               groups[int(g.group(1))] = metric['values'][0] if len(metric['values']) == 1 else metric['values']
    
    if args.g and len(groups) > 0:
        r = [None] * max(groups.keys())
    
        for  k,v in groups.items():
            r[k-1] = v
        print("Group {}: {}".format(args.g, r))


    result[res['_id']] = (expArgs, metricToPlot)


if args.m:
    colors = ['b', 'c', 'g', 'r', 'm', 'y', 'k', ]
    line = ['-', '--', '-.', ':']

    lineOpt = [c + l for c, l in itertools.product(colors, line)]

    random.seed(963261122)
    random.shuffle(lineOpt)

    plots = []

    for resId, (expArgs, metricsToPlot) in result.items():
        if len(metricsToPlot) == 0:
            continue

        metricValues = metricsToPlot[0]['values']
        if args.p:
            lab = ['{}={}'.format(k, v) for k, v in expArgs.items()]
            lab = ', '.join(lab) + ' ({})'.format(str(resId))
        else:
            lab = str(resId)

        plots.append(plt.plot(range(len(metricValues)), metricValues, lineOpt.pop(), label=lab)[0])

    labs = [l.get_label() for l in plots]
    plt.legend(plots, labs)
    plt.show()

if args.s:
    results_to_sort = []
    variable_arguments = []

    for resId, (expArgs, metricsToPlot) in result.items():
        if len(metricsToPlot) == 0:
            continue

        if args.p:
            lab = ['{}={}'.format(k, v) for k, v in expArgs.items()]
            lab = ', '.join(lab) + ' ({})'.format(str(resId))
        else:
            lab = expArgs

        results_to_sort.append((max(metricsToPlot[0]['values']),resId, lab,metricsToPlot[0]['values']))

    print("Top 50")
    for r in sorted(results_to_sort, reverse=True)[:50]:
        print("{}\t{}\t{}\t{}".format(r[2],r[0],r[1],r[3]))

