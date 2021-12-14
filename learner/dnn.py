import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import math
import conf
# from copy import deepcopy
# import random
# from sklearn.manifold import TSNE
# import seaborn as sns

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(conf.args.gpu_idx) # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator

class DNN():
    def __init__(self, model, dataloader, tensorboard, write_path):
        self.device = device
        self.tensorboard = tensorboard

        # init dataloader
        self.source_dataloader = dataloader
        self.write_path = write_path

        # init model
        self.model = model
        self.feature_extractor = model.Extractor().to(device)
        self.class_classifier = model.Class_Classifier().to(device)

        # init criterions
        self.class_criterion = nn.NLLLoss()
        # set hyperparameters
        # init self.optimizer
        self.optimizer = optim.Adam([{'params': self.feature_extractor.parameters()},
                                     {'params': self.class_classifier.parameters()}], lr=conf.args.opt['learning_rate'],
                                    weight_decay=conf.args.opt['weight_decay'])
        # self.optimizer = optim.SGD([{'params': self.feature_extractor.parameters()},
        #                              {'params': self.class_classifier.parameters()}], lr=conf.args.opt['learning_rate'],
        #                            weight_decay=conf.args.opt['weight_decay'])

        # learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=50,
                                                   gamma=0.1)

    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        state = {}
        state['epoch'] = epoch
        state['epoch_acc'] = epoch_acc
        state['best_acc'] = best_acc
        state['feature_extractor'] = self.feature_extractor.state_dict()
        state['class_classifier'] = self.class_classifier.state_dict()
        state['optimizer'] = self.optimizer.state_dict()

        torch.save(state, checkpoint_path)
        return

    def load_checkpoint(self, checkpoint_path, is_transfer=False):
        path = checkpoint_path
        checkpoint = torch.load(path, map_location=f'cuda:{conf.args.gpu_idx}')

        # FT_FC (FT_FC)
        if is_transfer:
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.class_classifier.load_state_dict(checkpoint['class_classifier'])
        # FT_all
        elif 'FT_all' in conf.args.method:
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            self.class_classifier.load_state_dict(checkpoint['class_classifier'])
        # resume training (Src, Tgt)
        else:
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            self.class_classifier.load_state_dict(checkpoint['class_classifier'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint

    def reset_layer(self):
        # Remember that Pytorch accumulates gradients.x
        # We need to clear them out before each instance
        # that means we have to call optimizer.zero_grad() before each backward.
        self.feature_extractor.zero_grad()
        self.class_classifier.zero_grad()

    def get_label_and_data(self, data):
        input_of_data, class_label_of_data = data
        input_of_data = input_of_data.to(device)
        class_label_of_data = class_label_of_data.to(device)

        return input_of_data, class_label_of_data

    def adjust_learning_rate(self, optimizer, epoch):
        """Decay the learning rate based on schedule"""
        lr = conf.args.cos_lr
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / conf.args.epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_feature(self, feature_extractor, data):
        return feature_extractor(data)

    def get_loss_and_confusion_matrix(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)

        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        return loss_of_data, cm, preds_of_data

    def get_loss_cm_error(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)
        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        assert (len(label) == len(pred_label))
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        errors = [0 if label[i] == pred_label[i] else 1 for i in range(len(label))]
        return loss_of_data, cm, errors

    def log_loss_results(self, condition, epoch, loss_avg):

        self.tensorboard.log_scalar(condition + '/loss_sum', loss_avg, epoch)

        # print loss
        print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(condition, epoch, loss_avg))

        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class):

        assert (condition in ['valid', 'test', 'demo'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)
        self.tensorboard.log_scalar(condition + '/' + 'accuracy_class_' + suffix, class_accuracy, epoch)

        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, suffix, class_accuracy))
        self.tensorboard.log_confusion_matrix(condition + '_accuracy_class_' + suffix, cm_class,
                                              conf.args.opt['classes'], epoch)

        return class_accuracy

    def train(self, epoch):
        """
        Train the model
        """

        # setup models

        self.feature_extractor.train()
        self.class_classifier.train()

        class_loss_sum = 0.0

        total_iter = 0

        if conf.args.method in ['Src', 'Src_Tgt']:
            num_iter = len(self.source_dataloader['train'])
            total_iter += num_iter
            for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['train']), total=num_iter):
                self.reset_layer()

                input_of_labeled_data, class_label_of_labeled_data = self.get_label_and_data(labeled_data)

                # compute the feature
                if conf.args.model == 'beat_change_model_lstm':
                    # reshape data from (batchsize, #feature, #sequence) to (batchsize, #sequence, #feature)
                    input_of_labeled_data = torch.reshape(input_of_labeled_data, (input_of_labeled_data.shape[0],
                                                                                  input_of_labeled_data.shape[2],
                                                                                  input_of_labeled_data.shape[1]))
                    self.feature_extractor.hidden = self.feature_extractor.init_hidden(batch_size=input_of_labeled_data.size(0))
                feature_of_labeled_data = self.get_feature(self.feature_extractor, input_of_labeled_data)

                # compute the class loss of feature_of_labeled_data
                class_loss_of_labeled_data, _, _ = self.get_loss_and_confusion_matrix(self.class_classifier,
                                                                                      self.class_criterion,
                                                                                      feature_of_labeled_data,
                                                                                      class_label_of_labeled_data)

                class_loss = class_loss_of_labeled_data
                class_loss_sum += float(class_loss * input_of_labeled_data.size(0))
                class_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        # if conf.args.method in ['Tgt', 'Src_Tgt', 'FT_FC', 'FT_all']:
        #
        #     self.reset_layer()
        #     total_iter += 1
        #
        #     input_of_labeled_data, class_label_of_labeled_data, _ = self.get_label_and_data(
        #         self.target_support_set)
        #
        #     # general case
        #     # compute the feature
        #     feature_of_labeled_data = self.get_feature(self.feature_extractor, input_of_labeled_data)
        #
        #     # compute the class loss of feature_of_labeled_data
        #     class_loss_of_labeled_data, _, _ = self.get_loss_and_confusion_matrix(self.class_classifier,
        #                                                                           self.class_criterion,
        #                                                                           feature_of_labeled_data,
        #                                                                           class_label_of_labeled_data)
        #
        #     class_loss = class_loss_of_labeled_data
        #     class_loss_sum += float(class_loss * input_of_labeled_data.size(0))
        #     class_loss.backward()
        #     if conf.args.grad_norm:
        #         self.get_grad_norm()
        #     self.optimizer.step()
        #
        #     if conf.args.motivation_reproduce:
        #         self.scheduler.step()
        #
        #     self.log_loss_results('train', epoch=epoch, loss_avg=class_loss_sum / total_iter)

        avg_loss = class_loss_sum / total_iter
        return avg_loss

    def logger(self, name, value, epoch, condition):

        if not hasattr(self, name + '_log'):
            exec(f'self.{name}_log = []')
            exec(f'self.{name}_file = open(self.write_path + name + ".txt", "w")')

        exec(f'self.{name}_log.append(value)')

        if isinstance(value, torch.Tensor):
            value = value.item()
        write_string = f'{epoch}\t{value}\n'
        exec(f'self.{name}_file.write(write_string)')
        # self.tensorboard.log_scalar(condition + '/' + name, value, epoch)

    def pca_analysis(self, epoch = 0, condition='test'):
        self.feature_extractor.eval()
        self.class_classifier.eval()

        class_loss_sum = 0.0
        class_cm_labeled_sum = 0
        class_cm_test_data_sum = 0

        num_iter = len(self.source_dataloader[condition])

        with torch.no_grad():
            for batch_idx, test_data in tqdm(enumerate(self.source_dataloader[condition]), total=num_iter):
                print("-----------")
                print(test_data[0].shape)
                print(test_data[1].shape)
                # test_data[0] = test_data[0].view(-1, *(test_data[0].shape[2:]))
                test_data[1] = test_data[1].view(-1)
                print(test_data[0].shape)
                print(test_data[1].shape)

                input_of_test_data, class_label_of_test_data = self.get_label_and_data(test_data)
                print(input_of_test_data.shape)
                print(class_label_of_test_data.shape)
                if conf.args.model == 'beat_change_model_lstm':
                    # reshape data from (batchsize, #feature, #sequence) to (batchsize, #sequence, #feature)
                    input_of_test_data = torch.reshape(input_of_test_data, (input_of_test_data.shape[0],
                                                                            input_of_test_data.shape[2],
                                                                            input_of_test_data.shape[1]))
                    self.feature_extractor.hidden = self.feature_extractor.init_hidden(batch_size=input_of_test_data.size(0))
                feature_of_test_data = self.get_feature(self.feature_extractor, input_of_test_data)
                # TODO: PCA analysis
                print("======================================================================")
                print(feature_of_test_data.shape)
                print(class_label_of_test_data.shape)
                x = pd.DataFrame(feature_of_test_data.numpy())
                y = pd.DataFrame(class_label_of_test_data.numpy())

                scaler = StandardScaler()
                x = scaler.fit_transform(x)

                pca = PCA(n_components=2)
                PCs = pca.fit_transform(x)

                principal_df = pd.DataFrame(data = PCs,
                                            columns = ['PC1', 'PC2'])

                final_df = pd.concat([principal_df, y], axis=1)
                final_df.columns = ['PC1','PC2','label']

                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(1,1,1)
                ax.set_xlabel('Principal Component 1', fontsize=15)
                ax.set_ylabel('Principal Component 2', fontsize=15)
                ax.set_title('2 component PCA', fontsize=20)

                labels = [0, 1, 2, 3]
                colors = ['r', 'b', 'g', 'c']
                for label, color in zip(labels, colors):
                    indices_to_keep = final_df['label'] == label
                    ax.scatter(final_df.loc[indices_to_keep, 'PC1'],
                               final_df.loc[indices_to_keep, 'PC2'],
                               c=color,
                               s=50,
                               alpha=0.2)
                # ax.legend(labels)
                ax.legend(['None', '3/4 beat 1', '3/4 beat 2', '3/4 beat3'])
                ax.grid()
                plt.show()
                plt.savefig('PCA.png', dpi=300)


    def demo_produce(self):
        self.feature_extractor.eval()
        self.class_classifier.eval()

        class_loss_sum = 0.0
        class_cm_labeled_sum = 0
        class_cm_test_data_sum = 0

        save_path = conf.args.load_checkpoint_path[:-3]
        if conf.args.type == 'beat_type':
            filename = 'beats_type.txt'
        elif conf.args.type == 'beat_change':
            filename = 'beats_change.txt'
        f = open(save_path+filename, 'w')

        num_iter = len(self.source_dataloader)
        with torch.no_grad():
            for batch_idx, test_data in tqdm(enumerate(self.source_dataloader), total=num_iter):
                input_of_test_data, class_label_of_test_data = self.get_label_and_data(test_data)

                if conf.args.model == 'beat_change_model_lstm':
                    # reshape data from (batchsize, #feature, #sequence) to (batchsize, #sequence, #feature)
                    input_of_test_data = torch.reshape(input_of_test_data, (input_of_test_data.shape[0],
                                                                            input_of_test_data.shape[2],
                                                                            input_of_test_data.shape[1]))
                    self.feature_extractor.hidden = self.feature_extractor.init_hidden(batch_size=input_of_test_data.size(0))
                feature_of_test_data = self.get_feature(self.feature_extractor, input_of_test_data)
                preds_of_data = self.class_classifier(feature_of_test_data)
                pred_label = preds_of_data.max(1, keepdim=False)[1]
                write_str = str(batch_idx) + " , " + str(pred_label.item()) + "\n"
                f.write(write_str)
                class_loss_of_test_data, class_cm_test_data, _ = self.get_loss_and_confusion_matrix(
                    self.class_classifier,
                    self.class_criterion,
                    feature_of_test_data,
                    class_label_of_test_data
                )
                class_loss_sum += float(class_loss_of_test_data * input_of_test_data.size(0))
                class_cm_test_data_sum += class_cm_test_data
        f.close()
        epoch_avg_loss = self.log_loss_results('demo', epoch=0, loss_avg=class_loss_sum / num_iter)
        class_accuracy_of_test_data = self.log_accuracy_results('demo', suffix='test', epoch=0,
                                                                cm_class=class_cm_test_data_sum)

    def evaluation(self, epoch, condition):

        self.feature_extractor.eval()
        self.class_classifier.eval()

        class_loss_sum = 0.0
        class_cm_labeled_sum = 0
        class_cm_test_data_sum = 0

        num_iter = len(self.source_dataloader[condition])

        with torch.no_grad():
            for batch_idx, test_data in tqdm(enumerate(self.source_dataloader[condition]), total=num_iter):
                # test_data[0] = test_data[0].view(-1, *(test_data[0].shape[2:]))
                test_data[1] = test_data[1].view(-1)

                input_of_test_data, class_label_of_test_data = self.get_label_and_data(test_data)

                if conf.args.model == 'beat_change_model_lstm':
                    # reshape data from (batchsize, #feature, #sequence) to (batchsize, #sequence, #feature)
                    input_of_test_data = torch.reshape(input_of_test_data, (input_of_test_data.shape[0],
                                                                            input_of_test_data.shape[2],
                                                                            input_of_test_data.shape[1]))
                    self.feature_extractor.hidden = self.feature_extractor.init_hidden(batch_size=input_of_test_data.size(0))
                feature_of_test_data = self.get_feature(self.feature_extractor, input_of_test_data)
                # class_loss_of_test_data, class_cm_test_data, _ = self.get_loss_and_confusion_matrix(
                #     self.class_classifier,
                #     self.class_criterion,
                #     feature_of_test_data,
                #     class_label_of_test_data)
                try:
                    class_loss_of_test_data, class_cm_test_data, _ = self.get_loss_and_confusion_matrix(
                                                                                    self.class_classifier,
                                                                                    self.class_criterion,
                                                                                    feature_of_test_data,
                                                                                    class_label_of_test_data)
                except:
                    print(feature_of_test_data.shape)

                class_loss_sum += float(class_loss_of_test_data * input_of_test_data.size(0))
                class_cm_test_data_sum += class_cm_test_data

        epoch_avg_loss = self.log_loss_results(condition, epoch=epoch, loss_avg=class_loss_sum / num_iter)
        class_accuracy_of_test_data = self.log_accuracy_results(condition, suffix='test', epoch=epoch, cm_class=class_cm_test_data_sum)
        self.logger('loss', epoch_avg_loss, epoch, condition)
        self.logger('accuracy', class_accuracy_of_test_data, epoch, condition)

        return class_accuracy_of_test_data, epoch_avg_loss, class_cm_test_data_sum

    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss, _ = self.evaluation(epoch, 'valid')

        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the performance of the model
        """

        #### for test data
        class_accuracy_of_test_data, loss, cm_class = self.evaluation(epoch, 'test')

        return class_accuracy_of_test_data, loss, cm_class
