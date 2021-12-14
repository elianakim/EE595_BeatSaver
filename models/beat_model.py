import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')

import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:{:d}".format(0) if torch.cuda.is_available() else "cpu")

# feature_flatten_dim = 2048
# feature_flatten_dim = 1664
if conf.args.type == "beat_change":
    feature_flatten_dim = 14848
elif conf.args.type == "beat_type":
    feature_flatten_dim = 66048
if conf.args.feat_eng:
    input_channel_dim = 12
else:
    input_channel_dim = 6


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv1d(input_channel_dim, 32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(32),


            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(64),


            nn.Conv1d(64, 128, kernel_size=3),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),


            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(256),


            nn.Conv1d(256, 512, kernel_size=3),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
        )

    def forward(self, input):
        out = self.feature(input)

        out = out.view(input.size(0), -1)
        return out

    def get_parameters(self):
        return [{"params": self.feature_extractor.parameters(), "lr_mult": 10, 'decay_mult': 2}]


class Class_Classifier(nn.Module):

    def __init__(self):
        super(Class_Classifier, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Linear(256, conf.args.opt['num_classes']))

    def forward(self, input):
        out = self.class_classifier(input)

        return F.log_softmax(out, 1)


    def get_logits(self, input):
        out = self.class_classifier(input)

        return out

    def get_parameters(self):
        return [{"params": self.class_classifier.parameters(), "lr_mult": 10, 'decay_mult': 2}]

if __name__ == '__main__':
    fe = Extractor()
    cc = Class_Classifier()

    input = torch.randn((10,6,200))
    clabel_tgt = cc(fe(input)) #[10,9]
    ## Target category diversity loss
    pb_pred_tgt = clabel_tgt.sum(dim=0) #[9]
    pb_pred_tgt = 1.0 / pb_pred_tgt.sum() * pb_pred_tgt#[9] sums to 1.   # normalization to a prob. dist.
    target_div_loss = - torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6)))

    target_entropy_loss = -torch.mean((clabel_tgt * torch.log(clabel_tgt + 1e-6)).sum(dim=1))
    print(target_div_loss)
    print(target_entropy_loss)
    pass