import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F

from const import *
from pdfa.symbol import SYMBOL


class CharacterClassificationModel(nn.Module):
    def __init__(self, input: list, encoder_hidden_sz: int, output: list, encoder_num_layers: int = 3, embed_sz: int = 16,
                 predictor_hidden_sz: int = 64, predictor_num_layers: int = 1, dropout: int = 0.1):
        """
        This module is made specifically for classifying characters in a name as first, middle, last, title 
        or suffix. Acts as a finite state automota by zeroing out moves that can't be taken
        """
        super(CharacterClassificationModel, self).__init__()
        self.encoder_hidden_sz = encoder_hidden_sz
        self.encoder_num_layers = encoder_num_layers
        self.predictor_hidden_sz = predictor_hidden_sz
        self.predictor_num_layers = predictor_num_layers
        self.input = input
        self.input_sz = len(input)
        self.predictor_input_sz = embed_sz * 4
        self.output = output
        self.output_sz = len(output)
        self.embed_sz = embed_sz
        self.input_pad_idx = self.input.index(PAD)
        self.softmax = nn.Softmax(dim=2)
        self.embed = nn.Embedding(self.input_sz, self.embed_sz, padding_idx=self.input_pad_idx)
        self.encoder_lstm = nn.LSTM(self.embed_sz, encoder_hidden_sz, num_layers=encoder_num_layers, bidirectional=True, dropout=dropout)
        self.predictor_lstm = nn.LSTM(self.predictor_input_sz, predictor_hidden_sz, num_layers=predictor_num_layers, dropout=dropout)
        self.fc1 = nn.Linear(self.encoder_hidden_sz * 2 + len(output), self.predictor_input_sz)
        self.fc2 = nn.Linear(self.predictor_hidden_sz, self.output_sz)

        self.rnn_encoder = None
        self.rnn_predictor = None

        self.to(DEVICE)
    
    def encode(self, input: torch.LongTensor):
        embedded_input = self.embed(input).unsqueeze(1)
        length_tensor = torch.Tensor([len(input)]).to(DEVICE)
        pps_input = torch.nn.utils.rnn.pack_padded_sequence(embedded_input, length_tensor)
        encoder_output, encoder_hidden = self.encoder_lstm(pps_input, self.init_encoder_hidden())
        encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_output)
        return encoder_output, encoder_hidden
    
    def predict(self, symbol: str, encoder_output: torch.Tensor, hidden_state: torch.Tensor):
        flattened_encoder_input = encoder_output.flatten()
        fc1_input = torch.cat((flattened_encoder_input, self.one_hot_encode(symbol, self.output)), dim=0)
        predictor_input = F.relu(self.fc1.forward(fc1_input))
        predictor_output, hidden_state = self.predictor_lstm(predictor_input.expand(1,1,self.predictor_input_sz), hidden_state)
        probs = self.softmax(self.fc2.forward(predictor_output))
        return probs, hidden_state

    def one_hot_encode(self, previous_sample: str, encoding_reference: list):
        ret = torch.zeros(len(encoding_reference)).to(DEVICE)
        ret[encoding_reference.index(previous_sample)] = 1
        return ret
    
    def test_mode(self):
        self.encoder_lstm.eval()
        self.predictor_lstm.eval()

    def init_encoder_hidden(self, batch_sz: int = 1):
        return (torch.zeros(self.encoder_num_layers * 2, batch_sz, self.encoder_hidden_sz).to(DEVICE),
                torch.zeros(self.encoder_num_layers * 2, batch_sz, self.encoder_hidden_sz).to(DEVICE))

    def init_predictor_hidden(self, batch_sz: int = 1):
        return (torch.zeros(self.predictor_num_layers, batch_sz, self.predictor_hidden_sz).to(DEVICE),
                torch.zeros(self.predictor_num_layers, batch_sz, self.predictor_hidden_sz).to(DEVICE))