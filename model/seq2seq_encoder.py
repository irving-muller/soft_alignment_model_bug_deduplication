import logging

import torch
import torch.nn as nn
from torch.nn import Dropout

from model.basic_module import RNNEncoder, lastVector, meanVector, maxVector, SortedRNNEncoder, sortInput, undoSortOuput

"""
The encoders encodes the candidate bugs using information from anchor bugs. 
"""


class Seq2SeqEncoder(nn.Module):

    def __init__(self, rnnType, embeddingObject, hiddenSize, numLayers, bidirectional, updateEmbedding, fixedSizeMethod,
                 dropout=0.0, cudaOn=False):
        super(Seq2SeqEncoder, self).__init__()

        self.cudaOn = cudaOn
        self.hiddenSize = hiddenSize
        self.isLSTM = rnnType == 'lstm'

        logging.getLogger(__name__).info("Seq2SeqEncoder: fixed_size_method=%s" % fixedSizeMethod)

        # Encode Anchor
        self.anchorEncoder = RNNEncoder(rnnType, embeddingObject, hiddenSize, numLayers, bidirectional, updateEmbedding,
                                        dropout=dropout, cudaOn=self.cudaOn, isLengthVariable=True)
        # Encode Candidate
        self.candidateEncoder = SortedRNNEncoder(rnnType, embeddingObject, hiddenSize, numLayers, bidirectional,
                                                 updateEmbedding, dropout=dropout, cudaOn=self.cudaOn)

        if fixedSizeMethod == 'last':
            self.fixedSizeFunc = lastVector
        elif fixedSizeMethod == 'mean':
            self.fixedSizeFunc = meanVector
        elif fixedSizeMethod == 'max':
            self.fixedSizeFunc = maxVector
        else:
            raise Exception('fixed size method {} is not valid.'.format(fixedSizeMethod))

        # This dropout is applied on the report embedding
        self.dropout = Dropout(dropout) if dropout > 0.0 else None

    def forward(self, anchorInput, anchorLength, candidateInput, candidateLength, batchSize):
        anchorEmb, anchorHidden = self.encodeAnchor(anchorInput, anchorLength, batchSize)
        candidateEmb = self.encodeCandidate(anchorHidden, batchSize, candidateInput, candidateLength)

        # # Mask the padding output
        # expandedLen = contenderInput.unsqueeze(1).expand(batchSize, self.hiddenSize)
        # mask = expandedLen < torch.arange(contenderInput.size()[1])

        return anchorEmb, candidateEmb

    def encodeCandidate(self, anchorHidden, batchSize, candidateInput, candidateLength):
        candidateOutputs, candidateHidden = self.candidateEncoder(candidateInput, anchorHidden, candidateLength,
                                                                  batchSize)
        candidateEmb = self.fixedSizeFunc(candidateOutputs, candidateLength)

        if self.dropout is not None:
            candidateEmb = self.dropout(candidateEmb)

        return candidateEmb

    def encodeAnchor(self, anchorInput, anchorLength, batchSize):
        anchorOutputs, anchorHidden = self.anchorEncoder(anchorInput, None, anchorLength, batchSize)
        anchorEmb = self.fixedSizeFunc(anchorOutputs, anchorLength)

        if self.dropout is not None:
            anchorEmb = self.dropout(anchorEmb)

        return anchorEmb, anchorHidden

    def getOutputSize(self):
        return self.hiddenSize * 2

    def encode(self, anchorInput, anchorLength, candidateInput, candidateLength, batchSize, anchorId, candidates,
               cache):
        cacheData = cache.getCache(anchorId, batchSize, self.isLSTM, self.cudaOn)

        if cacheData:
            anchorEmb, anchorHidden = cacheData
        else:
            anchorEmb, anchorHidden = self.encodeAnchor(anchorInput, anchorLength, batchSize)
            cache.setCache(anchorId, anchorEmb, anchorHidden, self.isLSTM)

        candidateEmb = self.encodeCandidate(anchorHidden, batchSize, candidateInput, candidateLength)

        return anchorEmb, candidateEmb


class ReplicatedEmbEncoder(nn.Module):
    """
    In this seq2seq, the decoder always receives the encoder embedding.
    """

    def __init__(self, rnnType, embeddingObject, hiddenSize, numLayers, bidirectional, updateEmbedding, fixedSizeMethod,
                 dropout=0.0, cudaOn=False):
        super(ReplicatedEmbEncoder, self).__init__()

        self.cudaOn = cudaOn
        self.hiddenSize = hiddenSize
        self.isLSTM = rnnType == 'lstm'
        self.numLayers = numLayers
        self.hiddenSize = hiddenSize
        self.bidirectional = bidirectional
        self.cudaOn = cudaOn
        self.dropout = dropout

        logging.getLogger(__name__).info("ReplicatedEmbEncoder: fixed_size_method=%s" % fixedSizeMethod)

        # Encode Anchor
        self.anchorEncoder = RNNEncoder(rnnType, embeddingObject, hiddenSize, numLayers, bidirectional, updateEmbedding,
                                        dropout=dropout, cudaOn=self.cudaOn, isLengthVariable=True)
        # Encode Candidate

        # Copy pre-trained embedding to the layer
        self.embedding = nn.Embedding(embeddingObject.getNumberOfVectors(), embeddingObject.getEmbeddingSize(),
                                      padding_idx=embeddingObject.getPaddingIdx())
        self.embedding.weight.data.copy_(torch.from_numpy(embeddingObject.getEmbeddingMatrix()))
        self.embedding.weight.requires_grad = updateEmbedding

        logging.getLogger(__name__).info(
            "RNN Encoder: type={}, hiddenSize={}, numLayers={}, update_embedding={}, bidirectional={}, dropout{}".format(
                self.type, self.hiddenSize, numLayers, updateEmbedding, bidirectional, dropout))

        if rnnType == 'lstm':
            self.rnnEncoder = nn.LSTM(embeddingObject.getEmbeddingSize() + hiddenSize, self.hiddenSize, self.numLayers,
                                      batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
        elif rnnType == 'gru':
            self.rnnEncoder = nn.GRU(embeddingObject.getEmbeddingSize() + hiddenSize, self.hiddenSize, self.numLayers,
                                     batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
        else:
            raise Exception('rnn type {} is not valid.'.format(rnnType))

        if fixedSizeMethod == 'last':
            self.fixedSizeFunc = lastVector
        elif fixedSizeMethod == 'mean':
            self.fixedSizeFunc = meanVector
        elif fixedSizeMethod == 'max':
            self.fixedSizeFunc = maxVector
        else:
            raise Exception('fixed size method {} is not valid.'.format(fixedSizeMethod))

        # This dropout is applied on the report embedding
        self.dropout = Dropout(dropout) if dropout > 0.0 else None

    def forward(self, anchorInput, anchorLength, candidateInput, candidateLength, batchSize):
        anchorEmb, anchorHidden = self.encodeAnchor(anchorInput, anchorLength, batchSize)
        candidateEmb = self.encodeCandidate(anchorEmb, anchorHidden, batchSize, candidateInput, candidateLength)

        # # Mask the padding output
        # expandedLen = contenderInput.unsqueeze(1).expand(batchSize, self.hiddenSize)
        # mask = expandedLen < torch.arange(contenderInput.size()[1])

        return anchorEmb, candidateEmb

    def encodeCandidate(self, anchorEmb, anchorHidden, batchSize, candidateInput, candidateLength):
        # Sort Batch ( we need to sort the anchorEmb too and resize it)
        x, initialHidden, sortedIdxs, lengths = sortInput(candidateInput, anchorHidden, candidateLength, self.isLSTM)
        sortedAnchorEmb = anchorEmb[sortedIdxs].unsqueeze(1).expand(batchSize, candidateInput.size()[1], -1)

        # Encode Step
        embedding = self.embedding(x)

        input = torch.cat((embedding, sortedAnchorEmb), dim=2)
        input = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)

        output, _ = self.rnnEncoder(input, initialHidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Unsort Batch
        candidateOutputs = undoSortOuput(output, lengths, sortedIdxs)

        # Fixed size representation
        candidateEmb = self.fixedSizeFunc(candidateOutputs, candidateLength)

        if self.dropout is not None:
            candidateEmb = self.dropout(candidateEmb)

        return candidateEmb

    def encodeAnchor(self, anchorInput, anchorLength, batchSize):
        anchorOutputs, anchorHidden = self.anchorEncoder(anchorInput, None, anchorLength, batchSize)
        anchorEmb = self.fixedSizeFunc(anchorOutputs, anchorLength)

        if self.dropout is not None:
            anchorEmb = self.dropout(anchorEmb)

        return anchorEmb, anchorHidden

    def getOutputSize(self):
        return self.hiddenSize * 2

    def encode(self, anchorInput, anchorLength, candidateInput, candidateLength, batchSize, anchorId, candidates,
               cache):
        cacheData = cache.getCache(anchorId, batchSize, self.isLSTM, self.cudaOn)

        if cacheData:
            anchorEmb, anchorHidden = cacheData
        else:
            anchorEmb, anchorHidden = self.encodeAnchor(anchorInput, anchorLength, batchSize)
            cache.setCache(anchorId, anchorEmb, anchorHidden, self.isLSTM)

        candidateEmb = self.encodeCandidate(anchorEmb, anchorHidden, batchSize, candidateInput, candidateLength)

        return anchorEmb, candidateEmb


class BahdanauEncoder(nn.Module):
    """
    Create two encoders. First encoder, called anchor, extract information from the anchor bug report.
    The second extracts feature from the candidate bug report using attention mechanism to
    receive information from first encoder

    Inspired on Bahdanau2015
    """

    def __init__(self, rnnType, embeddingObject, hiddenSize, numLayers, bidirectional, updateEmbedding,
                 attention, fixedSizeMethod, dropout=0.0, cudaOn=False):
        super(BahdanauEncoder, self).__init__()

        self.cudaOn = cudaOn
        self.attention = attention
        self.dropout = dropout
        self.hiddenSize = hiddenSize
        self.rnnType = rnnType
        self.bidirectional = bidirectional
        self.fixedSizeMethod = fixedSizeMethod

        logging.getLogger(__name__).info("BahdanauEncoder: fixed_size_method=%s" % fixedSizeMethod)

        dropoutHidden = dropout if numLayers > 1 else 0.0

        # Encode Anchor
        self.anchorEncoder = RNNEncoder(rnnType, embeddingObject, hiddenSize, numLayers, bidirectional, updateEmbedding,
                                        self.dropout, cudaOn=self.cudaOn, isLengthVariable=True)

        # Copy pre-trained embedding to the layer
        self.cadidateEmbedding = nn.Embedding(embeddingObject.getNumberOfVectors(), embeddingObject.getEmbeddingSize(),
                                              padding_idx=embeddingObject.getPaddingIdx())
        self.cadidateEmbedding.weight.data.copy_(torch.from_numpy(embeddingObject.getEmbeddingMatrix()))
        self.cadidateEmbedding.weight.requires_grad = updateEmbedding

        # Create RNN
        inputSize = embeddingObject.getEmbeddingSize() + hiddenSize
        self.isLSTM = self.rnnType == 'lstm'

        if rnnType == 'lstm':
            self.candidateRNN = nn.LSTM(inputSize, hiddenSize, numLayers,
                                        batch_first=True, bidirectional=bidirectional, dropout=dropoutHidden)
        elif rnnType == 'gru':
            self.candidateRNN = nn.GRU(inputSize, hiddenSize, numLayers,
                                       batch_first=True, bidirectional=bidirectional, dropout=dropoutHidden)
        else:
            raise Exception('rnn type {} is not valid.'.format(rnnType))

        if self.fixedSizeMethod == 'last':
            self.fixedSizeFunc = lastVector
        elif self.fixedSizeMethod == 'mean':
            self.fixedSizeFunc = meanVector
        elif self.fixedSizeMethod == 'max':
            self.fixedSizeFunc = maxVector
        else:
            raise Exception('fixed size method {} is not valid.'.format(fixedSizeMethod))

        # This dropout is applied on the report embedding
        self.dropout = Dropout(dropout) if dropout > 0.0 else None

    def encodeAnchor(self, anchorInput, anchorLength, batchSize):
        # Encode the anchor bug and generate a fixed representation
        anchorOutputs, anchorHidden = self.anchorEncoder(anchorInput, None, anchorLength, batchSize)
        anchorEmb = self.fixedSizeFunc(anchorOutputs, anchorLength)

        if self.dropout is not None:
            anchorEmb = self.dropout(anchorEmb)

        return anchorEmb, anchorOutputs, anchorHidden

    def encodeCandidate(self, candidateInput, candidateLength, batchSize, anchorOutputs, anchorHidden):
        candidateSeqMaxLen = candidateInput.shape[1]
        candidateOutputList = []

        previousHidden = anchorHidden

        # Unroll NN
        for stepIdx in range(candidateSeqMaxLen):
            """
            Get last hidden state of the candidate encoder. In the first iteration, we use the last hidden state 
            of the anchor encoder. As we can use LSTM's, we have to separate the cell state from hidden state. 
            """
            if self.isLSTM:
                previousCandidateHidden = previousHidden[0]
            else:
                previousCandidateHidden = previousHidden

            """
            The hidden state has the shape (num_layers * num_directions, batch, hidden_size) and
            we can't pass this vector direct to Attention Layer since it expects a vector with the following shape:
            (batch, decoder_seq_len, hidden_size). So, in order to use the Attention layer, we have to change the vector
            shape. First, we concatenate the output of last layer(bidirectional RNN have two last layers) and, then,
            we insert a sequence length dimension to the vector.     
            """
            if self.bidirectional:
                previousCandidateHidden = previousCandidateHidden[-2:].view(batchSize, 1, self.hiddenSize * 2)
            else:
                previousCandidateHidden = previousCandidateHidden[-1].view(batchSize, 1, self.hiddenSize)

            # Compute next step
            stepIn = candidateInput[:, stepIdx:stepIdx + 1]
            context, attention = self.attention(previousCandidateHidden, anchorOutputs)

            x = self.cadidateEmbedding(stepIn)
            _, stepHidden = self.candidateRNN(torch.cat((x, context), dim=2), previousHidden)

            """
            Mask the hidden output if length < step (Dealing batch with variable length)  
            (solution: https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py)
            """
            mask = (candidateLength < stepIdx).float().unsqueeze(1).expand(batchSize, self.hiddenSize)

            if self.isLSTM:
                nextHidden = stepHidden[0] * (1 - mask) + mask * previousHidden[0]
                nextCell = stepHidden[1] * (1 - mask) + mask * previousHidden[1]

                previousHidden = (nextHidden, nextCell)
            else:
                nextHidden = stepHidden * (1 - mask) + mask * previousHidden
                previousHidden = nextHidden

            # Mask the output to zero (get the hidden output of the last layer)
            stepOutput = nextHidden[-1] * (1 - mask)

            candidateOutputList.append(stepOutput)

        # generate a fixed representation
        candidateOutputs = torch.cat(candidateOutputList, dim=1).view(batchSize, candidateSeqMaxLen, self.hiddenSize)
        candidateEmb = self.fixedSizeFunc(candidateOutputs, candidateLength)

        if self.dropout is not None:
            candidateEmb = self.dropout(candidateEmb)

        return candidateEmb

    def forward(self, anchorInput, anchorLength, candidateInput, candidateLength, batchSize):
        anchorEmb, anchorOutputs, anchorHidden = self.encodeAnchor(anchorInput, anchorLength, batchSize)
        candidateEmb = self.encodeCandidate(candidateInput, candidateLength, batchSize, anchorOutputs, anchorHidden)

        return anchorEmb, candidateEmb

    def getOutputSize(self):
        return self.hiddenSize * 2

    def encode(self, anchorInput, anchorLength, candidateInput, candidateLength, batchSize, anchorId, candidates,
               cache):
        cacheData = cache.getCache(anchorId, batchSize, self.isLSTM, self.cudaOn)

        if cacheData:
            anchorEmb, anchorOutputs, anchorHidden = cacheData
        else:
            anchorEmb, anchorOutputs, anchorHidden = self.encodeAnchor(anchorInput, anchorLength, batchSize)
            cache.setCache(anchorId, anchorEmb, anchorOutputs, anchorHidden, self.isLSTM)

        candidateEmb = self.encodeCandidate(candidateInput, candidateLength, batchSize, anchorOutputs, anchorHidden)

        return anchorEmb, candidateEmb
