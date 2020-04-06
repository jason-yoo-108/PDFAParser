import torch
import torch.nn as nn
import torch.nn.functional as F

from const import *
from neural_net.cc_model import CharacterClassificationModel


class ClassificationDenoisingModel(CharacterClassificationModel):
    def __init__(self, input, encoder_hidden_sz, output, classifier_output, encoder_num_layers=3, embed_sz=16, 
                 predictor_hidden_sz=64, predictor_num_layers=1):
        """
        Extend CharacterClassificationModel with a MLP that does denoising classification
        """
        super().__init__(input, encoder_hidden_sz, output, encoder_num_layers=encoder_num_layers, embed_sz=embed_sz, 
                        predictor_hidden_sz=predictor_hidden_sz, predictor_num_layers=predictor_num_layers)
        self.flattened_hidden_state_len = self.encoder_hidden_sz * 2 * encoder_num_layers
        self.classifier_output_sz = len(classifier_output)
        self.fc3 = nn.Linear(encoder_hidden_sz * encoder_num_layers * 2 * 2, self.classifier_output_sz)
        self.fc4 = nn.Linear(self.classifier_output_sz, self.classifier_output_sz)
        self.classify_softmax = nn.Softmax(dim=0)
    
    def classify(self, hidden_state: torch.Tensor):
        joined_hidden_state = torch.cat((hidden_state[0], hidden_state[1]), dim=0)
        flattened_hidden_state = joined_hidden_state.flatten()
        output = self.fc3(flattened_hidden_state)
        output = F.relu(output)
        output = self.fc4(output)
        return self.classify_softmax(output)


class SequenceDenoisingModel(CharacterClassificationModel):
    def __init__(self, input, encoder_hidden_sz, output, decoder_output, encoder_num_layers=3, embed_sz=16, 
                 predictor_hidden_sz=64, predictor_num_layers=1):
        """
        Extend CharacterClassificationModel with a decoder that does denoising
        """
        super().__init__(input, encoder_hidden_sz, output, encoder_num_layers=encoder_num_layers, embed_sz=embed_sz, 
                        predictor_hidden_sz=predictor_hidden_sz, predictor_num_layers=predictor_num_layers)
        self.decoder_output = decoder_output
        self.decoder_output_sz = len(decoder_output)
        self.decoder_embed = nn.Embedding(self.input_sz, self.embed_sz, padding_idx=input.index(PAD))
        self.decoder_num_layers = encoder_num_layers * 2
        self.decoder_lstm = nn.LSTM(self.embed_sz, self.encoder_hidden_sz, num_layers=self.decoder_num_layers)
        self.fc3 = nn.Linear(self.encoder_hidden_sz, self.decoder_output_sz)
    
    def decode(self, character: str, hidden_state: torch.Tensor):
        embedded_input = self.decoder_embed(torch.LongTensor([self.input.index(character)]).to(DEVICE))
        decoder_output, hidden_state = self.decoder_lstm(embedded_input.unsqueeze(1), hidden_state)
        probs = self.softmax(self.fc3.forward(decoder_output))
        return probs, hidden_state

    def init_decoder_hidden(self, batch_sz: int = 1):
        return (torch.zeros(self.decoder_num_layers, batch_sz, self.encoder_hidden_sz).to(DEVICE),
                torch.zeros(self.decoder_num_layers, batch_sz, self.encoder_hidden_sz).to(DEVICE))
