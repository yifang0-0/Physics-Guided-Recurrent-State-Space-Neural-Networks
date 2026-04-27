import torch.utils.data
import pandas as pd
import os
import torch
import time
import sys
import matplotlib.pyplot as plt
import argparse
import numpy as np
import csv
import io
import subprocess
# os.chdir('../')
# sys.path.append(os.getcwd())
# import user-written files
import data.loader as loader
import training
import testing
import utils.datavisualizer as dv
from utils.utils import compute_normalizer
from utils.logger import set_redirects
from utils.utils import save_options
# import options files
import options.train_options as main_params
import options.train_options as model_params
import options.train_options as dynsys_params
import options.train_options as train_params
from models.model_state import ModelState


# %%####################################################################################################################
# Main function
########################################################################################################################
def run_main_50(options, path_general, file_name_general):
    print('Run file: main_50.py')
    start_time = time.time()
    print(time.strftime("%c"))

    # get correct computing device
    if torch.cuda.is_available():
        # get the usage of gpu memory from command "nvidia-smi" 
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        gpu_df = pd.read_csv(io.BytesIO(gpu_stats),names=['memory.used', 'memory.free'],skiprows=1)
        print('GPU usage:\n{}'.format(gpu_df))
        
        #get the id of the gpu with a maximum memory space left
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
        idx = gpu_df['memory.free'].astype(int).idxmax()        
        print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
        
        # run the task on the selected GPU
        torch.cuda.set_device(idx)
        device = torch.device('cuda') 
        gpu_name = torch.cuda.get_device_name(idx)
        print(f"Using GPU {idx}: {gpu_name}") 
    else:
        device = torch.device('cpu')
    print('Device: {}'.format(device))

    # get the options
    options['device'] = device
    options['dataset_options'] = dynsys_params.get_dataset_options(options['dataset'])
    options['model_options'] = model_params.get_model_options(options['model'], options['dataset'],
                                                              options['dataset_options'])
    options['train_options'] = train_params.get_train_options(options['dataset'])
    options['test_options'] = train_params.get_test_options()

    # print model type and dynamic system type
    print('\n\tModel Type: {}'.format(options['model']))
    print('\tDynamic System: {}\n'.format(options['dataset']))

    file_name_general = file_name_general + '_h{}_z{}_n{}'.format(options['model_options'].h_dim,
                                                                  options['model_options'].z_dim,
                                                                  options['model_options'].n_layers)
    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # set logger
    set_redirects(path, file_name_general)

    #set the number of evaluation rounds
    train_rounds = options["train_rounds"]
    start_from_round = options["start_from"]
    # print number of evaluations
    print('Total number of data point sets: {}'.format(train_rounds))

    # allocation
    all_vaf = torch.zeros([train_rounds])
    all_rmse = torch.zeros([train_rounds])
    all_likelihood = torch.zeros([train_rounds])
    all_df = {}

    for i in range(start_from_round, train_rounds):
        # print which training iteration this is
        print(' {}/{} round starts'.format(i+1,train_rounds))

        # Specifying datasets
        loaders = loader.load_dataset(dataset=options["dataset"],
                                    dataset_options=options["dataset_options"],
                                    train_batch_size=options["train_options"].batch_size,
                                    test_batch_size=options["test_options"].batch_size, 
                                    known_parameter=options["known_parameter"])

        # Compute normalizers
        if options["normalize"]:
            normalizer_input, normalizer_output = compute_normalizer(loaders['train'])
        else:
            normalizer_input = normalizer_output = None

        if options['known_parameter'] == 'B':
            options['model_options'].u_dim = 2
            
        # Define model
        modelstate = ModelState(seed=options["seed"],
                                nu=loaders["train"].nu, ny=loaders["train"].ny,
                                model=options["model"],
                                options=options,
                                normalizer_input=normalizer_input,
                                normalizer_output=normalizer_output)
        modelstate.model.to(options['device'])

        # save the options
        save_options(options, path_general, 'options.txt')
        
        
        if options['do_train']:
            print(' {}/{} model is being trained'.format(i+1,train_rounds))
            
            df = {}
            # train the model
            df = training.run_train(modelstate=modelstate,
                                    loader_train=loaders['train'],
                                    loader_valid=loaders['valid'],
                                    options=options,
                                    dataframe=df,
                                    path_general=path_general,
                                    file_name_general=file_name_general+str(i))
            df = pd.DataFrame(df)

        if options['do_test']:
            print(' {}/{} model is being tested'.format(i+1,train_rounds))
            # test the model
            df = {}
            df = testing.run_test(options, loaders, df, path_general, file_name_general+str(i))
            df = pd.DataFrame(df)
            
            
            # # test the model for 10 times
            # # make sure df is in dataframe format
            # df = pd.DataFrame({})
            # for i in range(10):
            #     df_single = {}
            #     # test the model
            #     df_single = testing.run_test(options, loaders, df_single, path_general, file_name_general)
            #     # make df_single a dataframe
            #     df_single = pd.DataFrame(df_single)
            #     df = df.append(df_single)
            
            
        # store values
        all_df[i] = df

        # save performance values
        # print(df['vaf'],df['vaf'][0],type(df['rmse']),type(df['vaf']))
        all_vaf[i] = df['vaf'][0]
        all_rmse[i] = df['rmse'][0]
        all_likelihood[i] = df['marginal_likeli'].item()
                
    # save data
    # get saving path
    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # to pandas
    all_df_list = []
    for _,i_df in all_df.items():
        all_df_list.append(i_df)
    all_df = pd.concat(all_df_list)
        
    print(all_df)
    # filename
    file_name = file_name_general + '_multitrain.csv'
    
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # # check if there's old log
    if os.path.exists(path + file_name):
        # read old data
        df_old = pd.read_csv(path + file_name,index_col=None)
        #append new to the old file
        df = df_old.append(df)
    
    # %% TODO: remember to decomment this section
    
    # save data
    all_df.to_csv(path_general + file_name)
    # save performance values
    torch.save(all_vaf, path_general + 'data/' + 'all_vaf.pt')
    torch.save(all_rmse, path_general + 'data/' + 'all_rmse.pt')
    torch.save(all_likelihood, path_general + 'data/' + 'all_likelihood.pt')
    
    # plot performance
    # train_rounds_idx = np.arange(1, 51)
    # dv.plot_perf_ndata(train_rounds_idx, all_vaf, all_rmse, all_likelihood, options, path_general)
    # %% TODO: End of this section
    # time output
    time_el = time.time() - start_time
    hours = time_el // 3600
    min = time_el // 60 - hours * 60
    sec = time_el - min * 60 - hours * 3600
    print('Total ime of file execution: {}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, min, sec))
    print(time.strftime("%c"))


# %%
# The `if __name__ == "__main__":` block is used to check if the current script is being run as the
# main program. If it is, then the code inside the block will be executed.

if __name__ == "__main__":
    # set (high level) options dictionary, if the basic options are expected from the augment parser, we set OPTION_SETTING_MANUALLY = True, else we change the options directly from the python file.
    OPTION_FROM_PARSER = True
    if OPTION_FROM_PARSER is True:
        options = {}

        main_params_parser = main_params.get_main_options()
        options['dataset'] = main_params_parser.dataset
        options['model'] = main_params_parser.model
        options['do_train'] = main_params_parser.do_train
        options['do_test'] = main_params_parser.do_test
        options['logdir'] = main_params_parser.logdir
        options['normalize'] = main_params_parser.normalize
        options['seed'] = main_params_parser.seed
        options['optim'] = main_params_parser.optim
        options['showfig'] = main_params_parser.showfig
        options['savefig'] = main_params_parser.savefig
        options['known_parameter'] = main_params_parser.known_parameter
        options['train_rounds'] = main_params_parser.train_rounds
        options['start_from'] = main_params_parser.start_from

        # print("Encountered errors loading the main options of the training/testing task")
        
        
    else:
        options = {
            'dataset': 'toy_lgssm',  # options: 'narendra_li', 'toy_lgssm', 'wiener_hammerstein'
            'model': 'VAE-RNN', # options: 'VAE-RNN', 'VRNN-Gauss', 'VRNN-Gauss-I', 'VRNN-GMM', 'VRNN-GMM-I', 'STORN'
            'do_train': True,
            'do_test': True,
            'logdir': 'multi_50',
            'normalize': True,
            'seed': 1234,
            'optim': 'Adam',
            'showfig': False,
            'savefig': True,
            'known_parameter': 'None',
            'train_rounds':50,
            'start_from':0
        }

    # get saving path
    path_general = os.getcwd() + '/log/{}/{}/{}_{}/'.format(options['logdir'],
                                                         options['dataset'],
                                                         options['model'],options['known_parameter'] )

    # get saving file names
    file_name_general = options['dataset']

    run_main_50(options, path_general, file_name_general)
