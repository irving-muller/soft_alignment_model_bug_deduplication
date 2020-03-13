import logging

import torch
from torch import nn as nn
from torch.nn import ModuleList, Sequential

from model.siamese import computeListOutputSize


class MultiLayerScorerPair(nn.Module):

    def __init__(self, encoders, hiddenSizes, hiddenActClass, outActClass=nn.Sigmoid, batchNormalization=False):
        super(MultiLayerScorerPair, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.encoders = ModuleList(encoders)
        encOutSize = computeListOutputSize(encoders)

        # self.logger.info(
        #     "Probability Pair NN: without_raw_bug={}, batch_normalization={}".format(self.withoutBugEmbedding,
        #                                                                              batchNormalization))
        seq = []
        last = encOutSize
        for currentSize in hiddenSizes:
            seq.append(nn.Linear(last, currentSize))

            if batchNormalization:
                seq.append(nn.BatchNorm1d(currentSize))

            seq.append(hiddenActClass())
            self.logger.info("==> Create Hidden Layer (%d,%d) in the classifier" % (last, currentSize))
            last = currentSize

        seq.append(nn.Linear(last, 1))

        if outActClass:
            seq.append(outActClass())

        self.sequential = Sequential(*seq)

    def getEncoders(self):
        return self.encoders

    def forward(self, encoderInputs):
        encoderOutputs = []

        for encoder, encoderInput in zip(self.encoders, encoderInputs):
            encoderOutputs.extend(encoder(*encoderInput))

        x = torch.cat(encoderOutputs, 1)

        return self.similarity(x)


    def similarity(self, x):
        x = self.sequential(x)

        return x.squeeze()

    def getOutputSize(self):
        return 1