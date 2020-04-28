import torch.nn as nn
import torch.nn.functional as F

from const import *


class PredefinedComponentsDenoiser(nn.Module):
    def __init__(self, input: list, hidden_sz: int, output: list, num_layers: int = 4, embed_sz: int = 16):
        super(PredefinedComponentsDenoiser, self).__init__()
        self.input = input
        self.input_sz = len(input)
        self.output = output
        self.output_sz = len(output)
        self.hidden_sz = hidden_sz
        self.num_layers = num_layers
        self.embed_sz = embed_sz

        self.embedder = nn.Embedding(self.input_sz, embed_sz, input.index(PAD))
        self.lstm = nn.LSTM(self.input_sz, hidden_sz,
                            num_layers, bidirectional=True)
        self.fc1_input_sz = 4 * num_layers + 2 * hidden_sz
        self.fc1 = nn.Linear(self.fc1_input_sz, hidden_sz)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_sz, self.output_sz)
        self.softmax = nn.Softmax(dim=0)
        self.to(DEVICE)

    def encode(self, input_component: str, hidden: torch.Tensor):
        if hidden is None:
            hidden = self.init_hidden()

        input_tensor = torch.LongTensor(
            [self.input.index(c) for c in input_component]).to(DEVICE)
        embedded_input = self.embedder(input_tensor)
        pps_input = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_input, len(input_component))
        output, hidden = self.lstm.forward(pps_input, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            encoder_output)

        return output, hidden

    def decode(self, hidden: torch.Tensor):
        flat_hidden = hidden.flatten()
        fc1_out = self.fc1.forward(flat_hidden)
        fc1_out = self.sigmoid(fc1_out)
        fc2_out = self.fc2.forward(fc1_out)
        probs = self.softmax(fc2_out)

        return probs

    def init_hidden(self):
        return (torch.zeros(self.num_layers * 2, 1, self.hidden_sz).to(DEVICE),
                torch.zeros(self.num_layers * 2, 1, self.hidden_sz).to(DEVICE))


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input: list, hidden_sz: int, output: list, num_layers: int = 3, embed_sz: int = 16):
        super(DenoisingAutoEncoder, self).__init__()
        self.encoder_input = input
        self.encoder_input_sz = len(input)
        self.decoder_output = output
        self.decoder_output_sz = len(output)

        # Embeddings
        self.encoder_embed = nn.Embedding(
            self.encoder_input_sz, embed_sz, padding_idx=input.index(
                PAD)
        )
        self.decoder_embed = nn.Embedding(
            self.decoder_output_sz, embed_sz, padding_idx=output.index(PAD))

        # LSTMs
        self.encoder_lstm = nn.LSTM(
            embed_sz, self.hidden_sz, num_layers=num_layers)
        self.decoder_lstm = nn.LSTM(
            self.embed_sz, self.hidden_sz, num_layers=num_layers)

        self.fc1 = nn.Linear(self.hidden_sz, self.decoder_output_sz)
        self.to(DEVICE)

    def encode(self, input: torch.LongTensor):
        embedded_input = self.encoder_embed(input).unsqueeze(1)
        pps_input = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_input)
        encoder_output, encoder_hidden = self.encoder_lstm(
            pps_input, self.init_hidden())
        encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            encoder_output)
        return encoder_output, encoder_hidden

    def decode(self, input_char: str, hidden_state: torch.Tensor):
        embedded_input = self.decoder_embed(
            torch.LongTensor([self.input.index(input)]).to(DEVICE))
        decoder_output, hidden_state = self.decoder_lstm(
            embedded_input.unsqueeze(1), hidden_state)
        probs = self.softmax(self.fc1.forward(decoder_output))
        return probs, hidden_state

    def init_hidden(self, batch_sz: int = 1):
        return (torch.zeros(self.decoder_num_layers, batch_sz, self.encoder_hidden_sz).to(DEVICE),
                torch.zeros(self.decoder_num_layers, batch_sz, self.encoder_hidden_sz).to(DEVICE))
