import numpy as np
import torch

from util.data_util import ReOrderedList
from time import time


class ExchangeInfoPairCollate(object):

    def __init__(self, inputHandlers, sortByHandlerIdx=None, labelDataType=None):
        self.inputHandlers = inputHandlers
        self.sortByHandlerIdx = sortByHandlerIdx
        self.labelDataType = labelDataType

    def collate(self, batch):
        """
        Prepare data before doing forward propagation and backpropagation.
        Dataset returns bug1 and bug2 and the pair label.
        bug1 and bug2 are lists which each dimension is related to a different information source of the bug.

        :param batch: batch of examples
        :return:
        """
        # For each information type, we will have a specific InputHandler
        inputBug1Batches = [[] for _ in self.inputHandlers]
        inputBug2Batches = [[] for _ in self.inputHandlers]
        labels = []

        # Separating X and Y
        for bug1, bug2, label in batch:
            # We put each information type in a same matrix. This matrix is the batch of a specific encoder.
            for infoIdx in range(len(self.inputHandlers)):
                inputBug1Batches[infoIdx].append(bug1[infoIdx])
                inputBug2Batches[infoIdx].append(bug2[infoIdx])

            labels.append(label)

        if self.sortByHandlerIdx is not None and len(batch) > 1:
            """
            We sort the batches using a certain input encoder. We assume that the input handler of this encoder has
            the attribute LENGTH_IDX which gives the position that the length of each sequence can be found.
            """
            # Get the input that will be used to sort the inputs
            inputHandlerToSort = self.inputHandlers[self.sortByHandlerIdx]
            inputBatchToSort = (inputBug1Batches[self.sortByHandlerIdx], inputBug2Batches[self.sortByHandlerIdx])
            encoderInputToSort = inputHandlerToSort.prepare(inputBatchToSort)

            # Get the correct order from a specific input
            lengths = encoderInputToSort[self.inputHandlers[self.sortByHandlerIdx].LENGTH_IDX]
            _, sortedIdxs = torch.sort(lengths, descending=True)
            sortedIdxs = sortedIdxs.data.cpu().numpy()

            # Sort inputs
            encoderInputs = [
                inputHandler.prepare((ReOrderedList(infoBug1, sortedIdxs), ReOrderedList(infoBug2, sortedIdxs))) for
                inputHandler, infoBug1, infoBug2 in zip(self.inputHandlers, inputBug1Batches, inputBug2Batches)]

            # Sort targets
            target = torch.tensor(labels[sortedIdxs], dtype=self.labelDataType)
        else:
            # Prepare the input to be send to a encoder
            encoderInputs = [inputHandler.prepare((infoBug1, infoBug2)) for inputHandler, infoBug1, infoBug2
                             in zip(self.inputHandlers, inputBug1Batches, inputBug2Batches)]

            # Transform labels to a tensor
            target = torch.tensor(labels, dtype=self.labelDataType)

        return (encoderInputs,), target


class LazyPairCollate(object):

    def __init__(self, fn_collate, preprocessor, negativeGenerator=None, model=None, loss=None):
        self.fn_collate = fn_collate
        self.preprocessor = preprocessor
        self.negativeGenerator = negativeGenerator
        self.loss = loss
        self.model = model

    def collate(self, batch):
        # Separating X and Y
        pair_inputs = []

        for bug1, bug2, label in batch:
            pair_inputs.append((self.preprocessor.extract(bug1), self.preprocessor.extract(bug2), label))

        if self.negativeGenerator:
            positive, negative = self.negativeGenerator.generatePairs(self.model, self.loss, pair_inputs, batch)
            pair_inputs = positive + negative

        return self.fn_collate(pair_inputs)


class PairBugCollate(object):

    def __init__(self, inputHandlers, labelDataType=None, ignore_target=False, unsqueeze_target=False):
        self.inputHandlers = inputHandlers
        self.labelDataType = labelDataType
        self.ignore_target = ignore_target
        self.unsqueeze_target = unsqueeze_target

    def to(self, batch, device):
        # num_workers > 0: tensors have the be transfer to the GPU in the main thread.
        x, target = batch
        new_x = []

        for bug_data in x:
            device_module_inputs = []
            for encoder_input in bug_data:
                device_nn_input = []
                for data in encoder_input:
                    if data is None:
                        device_nn_input.append(None)
                    else:
                        device_nn_input.append(data.to(device))
                device_module_inputs.append(device_nn_input)

            new_x.append(device_module_inputs)

        if self.unsqueeze_target:
            target = target.unsqueeze(1)

        if not self.ignore_target:
            target = target.to(device)

        return new_x, target

    def collate(self, batch):
        """
        Prepare data before doing forward propagation and backpropagation.
        Dataset returns bug1 and bug2 and the pair label.
        bug1 and bug2 are lists which each dimension is related to a different information source of the bug.

        :param batch: batch of examples
        :return:
        """
        # For each information type, we will have a specific InputHandler
        bug1InfoBatches = [[] for _ in self.inputHandlers]
        bug2InfoBatches = [[] for _ in self.inputHandlers]
        labels = []

        # Separating X and Y
        for bug1, bug2, label in batch:
            # We put each information type in a same matrix. This matrix is the batch of a specific enconder.
            for infoIdx, infoInput in enumerate(bug1):
                bug1InfoBatches[infoIdx].append(infoInput)

            for infoIdx, infoInput in enumerate(bug2):
                bug2InfoBatches[infoIdx].append(infoInput)

            labels.append(label)

        # Prepare the input to be send to a encoder
        query = [inputHandler.prepare(infoBatch) for inputHandler, infoBatch in
                 zip(self.inputHandlers, bug1InfoBatches)]
        candidate = [inputHandler.prepare(infoBatch) for inputHandler, infoBatch in
                     zip(self.inputHandlers, bug2InfoBatches)]

        # Transform labels to a tensor
        if self.ignore_target:
            target = None
        else:
            target = torch.tensor(labels, dtype=self.labelDataType)

        return (query, candidate), target


class TripletBugCollate(object):

    def __init__(self, inputHandlers):
        self.inputHandlers = inputHandlers

    def to(self, batch, device):
        # num_workers > 0: tensors have the be transfer to the GPU in the main thread.
        x, _ = batch
        new_x = []

        for bug_data in x:
            device_module_inputs = []
            for encoder_input in bug_data:
                device_nn_input = []
                for data in encoder_input:
                    if data is None:
                        device_nn_input.append(None)
                    else:
                        device_nn_input.append(data.to(device))
                device_module_inputs.append(device_nn_input)

            new_x.append(device_module_inputs)

        return new_x, None

    def collate(self, batch):
        """
        Prepare data before doing forward propagation and backpropagation.
        Dataset returns bug1 and bug2 and the pair label.
        bug1 and bug2 are lists which each dimension is related to a different information source of the bug.

        :param batch: batch of examples
        :return:
        """
        # For each information type, we will have a specific InputHandler
        anchorInfoBatches = [[] for _ in self.inputHandlers]
        posInfoBatches = [[] for _ in self.inputHandlers]
        negInfoBatches = [[] for _ in self.inputHandlers]
        labels = []

        # Separating X and Y
        for anchor, pos, neg in batch:
            # We put each information type in a same matrix. This matrix is the batch of a specific enconder.
            for infoIdx, infoInput in enumerate(anchor):
                anchorInfoBatches[infoIdx].append(infoInput)

            for infoIdx, infoInput in enumerate(pos):
                posInfoBatches[infoIdx].append(infoInput)

            for infoIdx, infoInput in enumerate(neg):
                negInfoBatches[infoIdx].append(infoInput)

        # Prepare the input to be send to a encoder
        anchorInput = [inputHandler.prepare(infoBatch) for inputHandler, infoBatch in
                       zip(self.inputHandlers, anchorInfoBatches)]
        posInput = [inputHandler.prepare(infoBatch) for inputHandler, infoBatch in
                    zip(self.inputHandlers, posInfoBatches)]
        negInput = [inputHandler.prepare(infoBatch) for inputHandler, infoBatch in
                    zip(self.inputHandlers, negInfoBatches)]

        return (anchorInput, posInput, negInput), None
