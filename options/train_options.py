import argparse
import torch
import numpy as np


def get_train_options(dataset_name):
    train_parser = argparse.ArgumentParser(description='training parameter')
    train_parser.add_argument('--clip', type=int, default=10, help='gradient clipping norm')
    train_parser.add_argument('--lr_scheduler_nstart', type=int, default=10, help='epoch to start lr scheduler')
    train_parser.add_argument('--print_every', type=int, default=1, help='print loss every N batches')
    train_parser.add_argument('--test_every', type=int, default=5, help='validate every N epochs')

    if dataset_name == 'cascaded_tank':
        train_parser.add_argument('--n_epochs', '--n_epoch', type=int, default=1000, help='number of training epochs')
        train_parser.add_argument('--init_lr', '--init_nr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-9, help='minimum learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=10, help='lr scheduler check interval')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='lr scheduler reduction factor')

    elif dataset_name == 'toy_lgssm_5_pre':
        train_parser.add_argument('--n_epochs', '--n_epoch', type=int, default=200, help='number of training epochs')
        train_parser.add_argument('--init_lr', '--init_nr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-7, help='minimum learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=30, help='lr scheduler check interval')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=5, help='lr scheduler reduction factor')

    elif dataset_name == 'industrobo':
        train_parser.add_argument('--n_epochs', '--n_epoch', type=int, default=200, help='number of training epochs')
        train_parser.add_argument('--init_lr', '--init_nr', type=float, default=1e-2, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-7, help='minimum learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=20, help='lr scheduler check interval')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='lr scheduler reduction factor')

    if torch.cuda.is_available():
        train_parser.add_argument('--batch_size', type=int, default=2048, help='batch size (GPU)')
    else:
        train_parser.add_argument('--batch_size', type=int, default=128, help='batch size (CPU)')

    train_options, unknown = train_parser.parse_known_args()
    return train_options


def get_test_options():
    test_parser = argparse.ArgumentParser(description='testing parameter')
    test_parser.add_argument('--batch_size', type=int, default=32)
    test_options, unknown = test_parser.parse_known_args()
    return test_options


def get_main_options():
    model_parser = argparse.ArgumentParser(description='Model Parameter')
    model_parser.add_argument('--dataset', metavar='', type=str, default='cascaded_tank')
    model_parser.add_argument('--model', metavar='', type=str, default='AE-RNN')
    model_parser.add_argument('--do_train', action="store_true")
    model_parser.add_argument('--do_test', action="store_true")
    model_parser.add_argument('--logdir', metavar='', type=str, default='default')
    model_parser.add_argument('--normalize', action='store_false', default=True)
    model_parser.add_argument('--seed', metavar='', type=int, default=1234)
    model_parser.add_argument('--optim', metavar='', type=str, default='Adam')
    model_parser.add_argument('--showfig', metavar='', type=bool, default=True)
    model_parser.add_argument('--savefig', metavar='', type=bool, default=True)
    model_parser.add_argument('--savelog', metavar='', type=bool, default=True)
    model_parser.add_argument('--saveoutput', metavar='', type=bool, default=True)
    model_parser.add_argument('--known_parameter', metavar='', type=str, default='None')
    model_parser.add_argument('--train_rounds', metavar='', type=int, default=1)
    model_parser.add_argument('--start_from', metavar='', type=int, default=0)

    model_options, unknown = model_parser.parse_known_args()
    return model_options


def get_system_options(dataset_name, dataset_options, train_options):
    if dataset_name == 'toy_lgssm_5_pre':
        system_parameter = {}
        system_parameter['A'] = np.array([[0.7, 0.8], [0, 0.1]])
        system_parameter['B'] = np.array([[-1], [0.1]])

        if dataset_options.A_prt_idx == 0:
            system_parameter['A_prt'] = np.array([[0, 0], [0, 0]])
        elif dataset_options.A_prt_idx == 1:
            system_parameter['A_prt'] = np.array([[0.6, 0.7], [-0.1, 0]])
        elif dataset_options.A_prt_idx == 2:
            system_parameter['A_prt'] = np.array([[0.7, 0.8], [0, 0.1]])

        if dataset_options.B_prt_idx == 0:
            system_parameter['B_prt'] = np.array([[0], [0]])
        elif dataset_options.B_prt_idx == 1:
            system_parameter['B_prt'] = np.array([[-1.1], [0]])
        elif dataset_options.B_prt_idx == 2:
            system_parameter['B_prt'] = np.array([[-1], [0.1]])

        if dataset_options.C_prt_idx == 0:
            system_parameter['C_prt'] = np.array([[0, 0]])
        elif dataset_options.C_prt_idx == 1:
            system_parameter['C_prt'] = np.array([[0.9, -0.1]])
        elif dataset_options.C_prt_idx == 2:
            system_parameter['C_prt'] = np.array([[1, 0]])

        system_parameter['sigma_state'] = np.sqrt(0.25)
        system_parameter['sigma_out'] = np.sqrt(1)

    elif dataset_name == 'industrobo':
        system_parameter = {}
        system_parameter['dt'] = dataset_options.dt
        system_parameter['roboname'] = dataset_options.roboname
        system_parameter['if_G'] = dataset_options.if_G == 1
        system_parameter['if_clip'] = dataset_options.if_clip == 1
        system_parameter['if_level2'] = dataset_options.if_level2 == 1
        system_parameter['if_bias'] = dataset_options.if_bias == 1
        system_parameter['if_level0'] = dataset_options.if_level0 == 1

    elif dataset_name == 'cascaded_tank':
        system_parameter = {}
        system_parameter['k1'] = np.array(0.0464)
        system_parameter['k2'] = np.array(0.0003)
        system_parameter['k3'] = np.array(0.0412)
        system_parameter['k4'] = np.array(0.0586)
        system_parameter['k5'] = np.array(0.0039)
        system_parameter['k6'] = np.array(0.0146)
        system_parameter['offset'] = np.array(0)
        system_parameter['x2Max'] = np.array(10)
        system_parameter['xMin'] = np.array(0)
        system_parameter['x1Max'] = np.array(10)
        system_parameter['dt'] = dataset_options.dt
    else:
        system_parameter = {}
    return system_parameter


def get_dataset_options(dataset_name):
    if dataset_name == 'cascaded_tank':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: cascaded tank')
        dataset_parser.add_argument('--y_dim', type=int, default=1)
        dataset_parser.add_argument('--u_dim', type=int, default=1)
        dataset_parser.add_argument('--seq_len_train', type=int, default=768)
        dataset_parser.add_argument('--seq_len_test', type=int, default=1024)
        dataset_parser.add_argument('--seq_len_val', type=int, default=256)
        dataset_parser.add_argument('--dt', type=float, default=4)
        dataset_parser.add_argument('--k_max_train', type=int, default=768)
        dataset_parser.add_argument('--k_max_test', type=int, default=1024)
        dataset_parser.add_argument('--k_max_val', type=int, default=256)
        dataset_options, unknown = dataset_parser.parse_known_args()

    elif dataset_name == 'toy_lgssm_5_pre':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: lgssm')
        dataset_parser.add_argument('--y_dim', type=int, default=1)
        dataset_parser.add_argument('--u_dim', type=int, default=1)
        dataset_parser.add_argument('--seq_len_train', type=int, default=64)
        dataset_parser.add_argument('--seq_len_test', type=int, default=None)
        dataset_parser.add_argument('--seq_len_val', type=int, default=64)
        dataset_parser.add_argument('--loss_type', type=int, default=0)
        dataset_parser.add_argument('--A_prt_idx', type=int, default=0)
        dataset_parser.add_argument('--B_prt_idx', type=int, default=0)
        dataset_parser.add_argument('--C_prt_idx', type=int, default=0)
        dataset_parser.add_argument('--k_max_train', type=int, default=2000)
        dataset_parser.add_argument('--k_max_test', type=int, default=5000)
        dataset_parser.add_argument('--k_max_val', type=int, default=2000)
        dataset_options, unknown = dataset_parser.parse_known_args()

    elif dataset_name == 'industrobo':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: industrobo')
        dataset_parser.add_argument('--y_dim', type=int, default=6)
        dataset_parser.add_argument('--u_dim', type=int, default=6)
        dataset_parser.add_argument('--input_channel', type=int, default=1)
        dataset_parser.add_argument('--dt', type=float, default=0.1)
        dataset_parser.add_argument('--if_clip', type=int, default=1)
        dataset_parser.add_argument('--if_G', type=int, default=1)
        dataset_parser.add_argument('--if_level2', type=int, default=1)
        dataset_parser.add_argument('--if_bias', type=int, default=0)
        dataset_parser.add_argument('--if_level0', type=int, default=1)
        dataset_parser.add_argument('--if_simulation', type=int, default=0)
        dataset_parser.add_argument('--roboname', type=str, default="KUKA300")
        dataset_parser.add_argument('--seq_stride', type=int, default=None)
        dataset_parser.add_argument('--seq_len_train', type=int, default=606)
        dataset_parser.add_argument('--seq_len_test', type=int, default=606)
        dataset_parser.add_argument('--seq_len_val', type=int, default=606)
        dataset_parser.add_argument('--k_max_train', type=float, default=1)
        dataset_parser.add_argument('--k_max_test', type=int, default=3636)
        dataset_parser.add_argument('--k_max_val', type=int, default=3998)
        dataset_options, unknown = dataset_parser.parse_known_args()

    return dataset_options


def get_model_options(model_type, dataset_name, dataset_options):
    y_dim = dataset_options.y_dim
    u_dim = dataset_options.u_dim

    model_parser = argparse.ArgumentParser(description='Model Parameter')
    model_parser.add_argument('--y_dim', type=int, default=y_dim)
    model_parser.add_argument('--u_dim', type=int, default=u_dim)
    model_parser.add_argument('--x_phy_w', type=float, default=1)
    model_parser.add_argument('--x_nn_w', type=float, default=1)

    if dataset_name == 'cascaded_tank':
        model_parser.add_argument('--h_dim', type=int, default=60)
        model_parser.add_argument('--z_dim', type=int, default=2)
        model_parser.add_argument('--n_layers', type=int, default=1)

    elif dataset_name == 'toy_lgssm_5_pre':
        model_parser.add_argument('--h_dim', type=int, default=10)
        model_parser.add_argument('--z_dim', type=int, default=2)
        model_parser.add_argument('--n_layers', type=int, default=1)

    elif dataset_name == 'industrobo':
        model_parser.add_argument('--h_dim', type=int, default=64)
        model_parser.add_argument('--z_dim', type=int, default=12)
        model_parser.add_argument('--n_layers', type=int, default=3)

    model_parser.add_argument('--mpnt_wt', type=float, default=0)

    model_options, unknown = model_parser.parse_known_args()
    return model_options
