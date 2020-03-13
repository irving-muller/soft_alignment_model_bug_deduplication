"""
In this module, there are NN that use siamese neural network and receive pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basic_module import TextCNN


class DBR_CNN(nn.Module):
    """
    Implementation of Detecting Duplicate Bug Reports with Convolutional Neural Networks. Qi Xie. 2018
    """

    def __init__(self, word_embedding, window_sizes, n_filters, update_embedding):
        super(DBR_CNN, self).__init__()
        self.textual_encoder = TextCNN(window_sizes, n_filters, word_embedding, update_embedding, activationFunc=torch.tanh)
        self.classifier = nn.Linear(4, 1)

    def encode(self, textual_features):
        return self.textual_encoder(*textual_features)

    def forward(self, query, candidate):
        """
        :param inputs: (batch, seq_len) tensor containing word indexes for each example
        :return: (batch, num_classes) tensor containing scores for each class
        """
        query_categorical = query[0]
        query_textual = query[1]

        candidate_categorical = candidate[0]
        candidate_textual = candidate[1]

        query_emb = self.encode(query_textual)
        candidate_emb = self.encode(candidate_textual)

        return self.similarity(query_categorical, query_emb, candidate_categorical, candidate_emb)

    def similarity(self, query_categorical, query_emb, candidate_categorical, candidate_emb):
        query_component, query_priority, query_create_time = query_categorical
        cand_component, cand_priority, cand_create_time = candidate_categorical

        # Like Sun2011
        diff_component = (query_component == cand_component).float()
        diff_priority = 1.0 / (1.0 + torch.abs(query_priority - cand_priority))
        diff_create = 1.0 / (1.0 + torch.abs(query_create_time - cand_create_time))

        # Cosine similarity between textual features
        cos_sim = F.cosine_similarity(query_emb, candidate_emb).unsqueeze(1)

        # Concatenate features and classify
        ftrs = torch.cat([diff_component, diff_priority, diff_create, cos_sim], 1)

        return torch.sigmoid(self.classifier(ftrs))
