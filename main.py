import argparse
import sys
import os
import random
import time

import numpy as np
import torch
from tensorboard_logger import Tensorboard

import conf

def get_path():
    path = 'log/'

    # info about data
    path += conf.args.dataset + '/'

    # info about method
    path += conf.args.method + '/'

    path += conf.args.log_suffix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path + '/'
    result_path = path + '/'

    print('Path: {}'.format(path))
    return result_path, checkpoint_path, log_path


def main():

    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    ################## Load default hyperparameter ##################
    if 'beat_change' in conf.args.type:
        opt = conf.BeatChange_Opt
    elif 'beat_type' in conf.args.type:
        opt = conf.BeatType_Opt

    if conf.args.lr:
        opt['learning_rate'] = conf.args.lr
    conf.args.opt = opt

    ################## Load model ##################
    model = None
    if conf.args.model == 'beat_change_model':
        import models.beat_model as model
    elif conf.args.model == 'beat_type_model':
        import models.beat_model as model

    ################## Load dataset ##################
    from data_loader import data_loader as data_loader
    from learner.dnn import DNN
    result_path, checkpoint_path, log_path = get_path()
    tensorboard = Tensorboard(log_path)

    if conf.args.method in ['Src']:
        dataloader = data_loader.single_domain_data_loader(conf.args.opt['file_path'],
                                                            batch_size=conf.args.opt['batch_size'],
                                                            valid_split=0.1,
                                                            test_split=0.1)
        learner = DNN(model, dataloader=dataloader, tensorboard=tensorboard, write_path=log_path)

    ################## Training ##################
    if not conf.args.test_only:
        since = time.time()

        start_epoch = 1
        best_acc = -9999
        best_epoch = -1
        test_epoch = 1

        for epoch in range(start_epoch, conf.args.epoch + 1):
            if epoch == start_epoch:
                # make dir if doesn't exist
                if not os.path.exists(result_path):
                    oldumask = os.umask(0)
                    os.makedirs(result_path, 0o777)
                    os.umask(oldumask)
                if not os.path.exists(checkpoint_path):
                    oldumask = os.umask(0)
                    os.makedirs(checkpoint_path, 0o777)
                    os.umask(oldumask)
                # tensorboard
                for arg in vars(conf.args):
                    tensorboard.log_text('args/' + arg, getattr(conf.args, arg), 0)
                script = ' '.join(sys.argv[1:])
                tensorboard.log_text('args/script', script, 0)

                # initial validation
                epoch_acc, epoch_loss, _ = learner.test(0)
                if (epoch_acc >= best_acc):
                    best_acc = epoch_acc
                    best_epoch = epoch
                    learner.save_checkpoint(epoch=0, epoch_acc=epoch_acc, best_acc=best_acc,
                                            checkpoint_path=checkpoint_path + 'cp_best.pth.tar')
            learner.train(epoch)

            if (epoch % test_epoch == 0):
                epoch_acc, epoch_loss, _ = learner.test(epoch)
                # keep best accuracy and model
                if (epoch_acc > best_acc):
                    best_acc = epoch_acc
                    best_epoch = epoch
                    learner.save_checkpoint(epoch=epoch, epoch_acc=epoch_acc, best_acc=best_acc,
                                            checkpoint_path=checkpoint_path + 'cp_best.pth.tar')

        time_elapsed = time.time() - since

        learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                checkpoint_path=checkpoint_path + 'cp_last.pth.tar')

        print('Training complete time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

    else: # only for test
        resume = conf.args.load_checkpoint_path + 'cp_best.pth.tar'
        checkpoint = learner.load_checkpoint(resume)
        if checkpoint is not None:
            epoch_acc, epoch_loss, cm_class = learner.test(0)
            print(cm_class)

def parse_arguments(argv):
    """
    Parse a command line.
    """
    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    ### MANDATORY ###
    parser.add_argument('--type', type=str, default='',
                        help='Type of the model to train, between "beat_change" and "beat_type".')
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset to be used, in [beat_original].')
    parser.add_argument('--model', type=str, default='beat_model',
                        help='Model to be used.')
    parser.add_argument('--method', type=str, default='',
                        help='Src')
    parser.add_argument('--test_only', action='store_true',
                        help='For test only, without training.')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='which gpu to use')

    ### OPTIONAL ###
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate to overwrite conf.py')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--epoch', type=int, default=200,
                        help='How many epochs do you want to use for train')
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Load checkpoint and train from checkpoint in path?')
    parser.add_argument('--log_suffix', type=str, default='',
                        help='Suffix of log file path')
    parser.add_argument('--remove_cp', action='store_true',
                        help='Remove checkpoints after evaluation')

    return parser.parse_args()

def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__=='__main__':
    conf.args = parse_arguments(sys.argv[1:])
    set_seed()
    main()