import pymongo
from datetime import datetime
import sys


def preprocessMasterNotInDB():
    nChoseNewMaster = 0
    nChangedResolution = 0
    nK = 0

    for masterId in masterIdList:
        # Check if master exists in the database
        if masterId in bugs:
            continue

        # Remove master set of master id
        masterSet = masterSets.pop(masterId)
        masters.remove(masterId)

        # If master set has only one duplicate bug, so set the bug resolution to NDuplicate and dup_id to []
        if len(masterSet) == 1:
            bugId = masterSet.pop()
            bug = bugs[bugId]

            bug['resolution'] = 'NDUPLICATE'
            bug['dup_id'] = []

            # Bug is a master
            masters.add(bugId)
            # Bug doesn't have a master id
            masterByBug.pop(bugId)

            nChangedResolution += 1
            nK += 1
        else:
            # Consider the oldest bug as the new master of the group
            newMasterId = None
            oldestDate = datetime.now()

            for b in masterSet:
                creationDate = datetime.strptime(bugs[b]['creation_ts'].rsplit(' ', 1)[0], '%Y-%m-%d %H:%M:%S')

                if creationDate < oldestDate:
                    oldestDate = creationDate
                    newMasterId = b

            masterSet.remove(newMasterId)
            masterSets[newMasterId] = masterSet
            masters.add(newMasterId)

            masterBug = bugs[newMasterId]
            masterBug['dup_id'] = []
            masterBug['resolution'] = 'DMASTER'

            nK += 1

            for b in masterSet:
                dupBug = bugs[b]
                masterByBug[b] = newMasterId
                dupBug['dup_id'] = newMasterId
                nK += 1

            nChoseNewMaster += 1

    print("Amount of masters that were not in the database: %d" % (nChoseNewMaster + nChangedResolution))
    print("Amount of times that the master set had only 1 bug: %d" % (nChangedResolution))
    print("Amount of times that we picked a new master from the master set: %d" % (nChoseNewMaster))


if __name__ == '__main__':
    client = pymongo.MongoClient()
    database = client[sys.argv[1]]
    originalCollection = database[sys.argv[2]]
    newCollection = database[sys.argv[3]]

    bugs = {}
    masterByBug = {}
    masterSets = {}
    masters = set()
    openedStatus = ['UNCONFIRMED', 'NEW', 'ASSIGNED', 'REOPENED', 'READY']

    nRepeated = 0
    nOpenedBugs = 0
    dupIds = set()

    for bug in originalCollection.find({}):
        bugId = bug['bug_id']

        del bug['_id']

        if bugId in bugs:
            # Remove repeated bugs
            nRepeated += 1
            continue

        status = bug['bug_status']

        if status in openedStatus:
            # Remove bug that has opened status
            nOpenedBugs += 1
            continue

        bugs[bugId] = bug
        dupId = bug['dup_id']

        if dupId is not None and len(dupId) != 0:
            # set to 'DUPLICATE' the resolution
            bug['resolution'] = 'DUPLICATE'

            masterByBug[bugId] = dupId
            masters.add(dupId)

            l = masterSets.get(dupId, set())

            if len(l) == 0:
                masterSets[dupId] = l

            l.add(bugId)
        else:
            if bug['resolution'] == 'DUPLICATE':
                bug['resolution'] = 'NDUPLICATE'

            dupId = None
            masters.add(bugId)

    print("Amount of repeated bugs: %d" % nRepeated)
    print("Amount of opened bugs: %d" % nOpenedBugs)

    '''
    Guarantee that every duplicate bug has your master in the database.
    '''
    masterIdList = [i for i in masterSets.keys()]
    nMasterNotInDB = 0
    nModDuplicate = 0

    for masterId in masterIdList:
        # Check if master exists in the database
        if masterId in bugs:
            continue

        # Remove master set of master id
        masterSet = masterSets.pop(masterId)
        masters.remove(masterId)

        # If master set has only one duplicate bug, so set the bug resolution to NDuplicate and dup_id to []
        for bugId in masterSet:
            bug = bugs[bugId]

            bug['resolution'] = 'NDUPLICATE'
            bug['dup_id'] = []

            # Bug is a master
            masters.add(bugId)
            # Bug doesn't have a master id
            masterByBug.pop(bugId)

            nModDuplicate += 1

        nMasterNotInDB += 1

    print("Amount of masters that were not in the database: %d" % (nMasterNotInDB))
    print("Amount of duplicate bug that your resolution were modified to NDuplicate: %d" % (nModDuplicate))

    del masterIdList

    # Remove cycle
    cpItems = [(bugId, masterId) for bugId, masterId in masterByBug.items()]
    nCycles = 0
    for bugId, masterId in cpItems:
        masterIdOfMaster = masterByBug.get(masterId, -1)

        if masterIdOfMaster == bugId:
            creationDateBug = datetime.strptime(bugs[bugId]['creation_ts'].rsplit(' ', 1)[0], '%Y-%m-%d %H:%M:%S')
            creationDateMaster = datetime.strptime(bugs[masterId]['creation_ts'].rsplit(' ', 1)[0], '%Y-%m-%d %H:%M:%S')

            if creationDateBug < creationDateMaster:
                newMasterId = bugId
                oldMasterId = masterId
            else:
                newMasterId = masterId
                oldMasterId = bugId

            bug = bugs[newMasterId]
            bug['dup_id'] = []
            bug['resolution'] = 'DUPLICATE'

            masters.remove(oldMasterId)
            masterByBug.pop(oldMasterId)
            masterSets.pop(oldMasterId)

            nCycles += 1

    print("Amount of cycles removed: %d" % (nCycles))

    # Guarantee that the masters doesn't have a dup_id
    nMerge = 0

    for masterId in list(masterSets.keys()):
        # if A is duplicate of B and B is duplicate of C, so A is duplicate of C
        nextMasterId = masterByBug.get(masterId, None)

        if nextMasterId is None:
            continue

        parentId = masterId

        while nextMasterId is not None:
            parentId = nextMasterId
            nextMasterId = masterByBug.get(parentId, None)

        masters.remove(masterId)

        masterSet = masterSets.pop(masterId)
        masterSet.add(masterId)

        for bugId in masterSet:
            bug = bugs[bugId]
            bug['dup_id'] = parentId
            masterByBug[bugId] = parentId

        masterSets[parentId] = masterSets[parentId] | masterSet

        nMerge+=1

    print('We merged %d master bugs that dup_id is not empty with the true master' % nMerge)

    print("Saving bugs in new collection")
    newCollection.insert_many(bugs.values())