import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')

import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

feature_flatten_dim = 14848
if conf.args.feat_eng:
    input_channel_dim = 12
else:
    input_channel_dim = 6

class Extractor(nn.Module): # LSTM extractor

    def __init__(self):
        super(Extractor, self).__init__()

        self.hidden_dim = conf.BeatChange_LSTM_Opt['hidden_dim']
        self.num_layers = conf.BeatChange_LSTM_Opt['num_lstm_layers']
        self.num_directions = 2 if conf.BeatChange_LSTM_Opt['bidirectional'] else 1
        self.hidden = self.init_hidden()

        self.feature = nn.LSTM(
            input_channel_dim,
            self.hidden_dim,
            num_layers = self.num_layers,
            bias = True,
            batch_first = True,
            bidirectional = conf.BeatChange_LSTM_Opt['bidirectional'],
        )

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(device))

    def forward(self, input):
        out, self.hidden = self.feature(input, self.hidden)
        return out

    def get_parameters(self):
        return [{"params": self.feature_extractor.parameters(), "lr_mult": 10, "decay_mult": 2}]

class Class_Classifier(nn.Module):
    def __init__(self):
        super(Class_Classifier, self).__init__()

        self.hidden_dim = conf.BeatChange_LSTM_Opt['hidden_dim']
        self.num_layers = conf.BeatChange_LSTM_Opt['num_lstm_layers']
        self.num_directions = 2 if conf.BeatChange_LSTM_Opt['bidirectional'] else 1
        self.feature_flatten_dim = self.hidden_dim * self.num_directions

        self.class_classifier = nn.Sequential(
            nn.Linear(self.feature_flatten_dim, 30),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Linear(30, conf.BeatChange_LSTM_Opt['num_classes'])
        )

    def forward(self, input):
        out_fc = []
        for frame in input:
            out = F.log_softmax(self.class_classifier(frame)[-1])
            out_fc.append(out)
        return torch.stack(out_fc)

    def get_logits(self, input):
        out = self.class_classifier(input)
        return out

    def get_parameters(self):
        return [{"params": self.class_classifier.parameters(), "lr_mult":10, "decay_mult": 2}]

if __name__=='__main__':
    with torch.no_grad():
        dummy_input = torch.zeros((32, 12, 133)).to(device)
        ideal_input = torch.zeros(32, 133, 12) #        (batch, seq, feature)

