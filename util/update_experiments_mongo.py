import datetime
import json
import logging
import os
import sys
from time import time

import pymongo

folder = sys.argv[1]
mongoDB = sys.argv[2]

client = pymongo.MongoClient()
database = client[mongoDB]

runCollection = database['runs']
metricCollection = database['metrics']

# Get last update
updateCollection = database['last_update']

if updateCollection.count() == 0:
    lastUpdate = {'time': time()}
    lastUpdate['_id'] = updateCollection.insert_one(lastUpdate).inserted_id
else:
    lastUpdate = updateCollection.find_one()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
lastTime = lastUpdate['time']

logger.info("Last update: {}".format(lastTime))

for exp_folder in os.listdir(folder):
    expFolderPath = os.path.join(folder, exp_folder)

    if not os.path.isdir(expFolderPath):
        continue

    modifyTime = os.path.getmtime(expFolderPath)

    if modifyTime < lastTime:
        continue

    files = os.listdir(expFolderPath)

    if 'run.json' not in files:
        continue

    if os.path.getsize(os.path.join(expFolderPath, 'run.json')) <= 0:
        continue

    runJson = json.load(open(os.path.join(expFolderPath, 'run.json')))

    configJson = json.load(open(os.path.join(expFolderPath, 'config.json')))
    print(os.path.join(expFolderPath, 'metrics.json'))

    metricsJsonFilePath = os.path.join(expFolderPath, 'metrics.json')

    if os.path.isfile(metricsJsonFilePath) and  os.path.getsize(metricsJsonFilePath) > 0:
        metricsJson = json.load(open(metricsJsonFilePath))
    else:
        metricsJson = {}

    runJson['config'] = configJson
    runId = int(exp_folder)
    runJson['captured_out'] = os.path.join(expFolderPath, 'cout.txt')

    if os.path.isfile(os.path.join(expFolderPath, 'info.json')):
        runJson['info'] = json.load(open(os.path.join(expFolderPath, 'info.json')))
    else:
        runJson['info'] = {}

    for k in ["start_time", "stop_time", "heartbeat"]:
        if k in runJson and runJson[k] is not None:
            runJson[k] = datetime.datetime.strptime(runJson[k],
                                                    '%Y-%m-%dT%H:%M:%S.%f')

    metricsObjList = []

    for metricName, metricObj in metricsJson.items():
        metricObj['name'] = metricName
        metricObj['run_id'] = runId

        metricObj['timestamps'] = [datetime.datetime.strptime(timestamp,
                                                              '%Y-%m-%dT%H:%M:%S.%f') for timestamp in
                                   metricObj['timestamps']]

        metricsObjList.append(metricObj)

    if runCollection.find({'_id': runId}).count() > 0:
        # Update
        logger.info('Update information of {}'.format(runId))
        metricsInDb = {}

        for oldMetricObj in metricCollection.find({'run_id': runId}):
            metricsInDb[oldMetricObj['name']] = oldMetricObj['_id']

        metricsInfo = []

        for metricObj in metricsObjList:
            metricId = metricsInDb.get(metricObj['name'])

            if metricId:
                metricCollection.update_one({'_id': metricId}, {"$set": metricObj})
            else:
                metricId = metricCollection.insert_one(metricObj).inserted_id

            metricsInfo.append({"id": str(metricId), "name": metricObj['name']})

        runJson['info']['metrics'] = metricsInfo
        runCollection.update_one({'_id': runId}, {"$set": runJson})
    else:
        # Save
        logger.info('Save information of {}'.format(runId))
        runJson['_id'] = int(exp_folder)

        if len(metricsObjList) > 0:
            metricsInfo = []
            metricIds = metricCollection.insert_many(metricsObjList, ordered=True).inserted_ids

            assert len(metricIds) == len(metricsObjList)

            metricsInfo = [{"id": str(metricId), "name": metricObj['name']} for metricObj, metricId in
                           zip(metricsObjList, metricIds)]

            runJson['info']['metrics'] = metricsInfo

        runCollection.insert_one(runJson)

updateCollection.update_one({'_id': lastUpdate['_id']}, {"$set": {'time': time()}})
