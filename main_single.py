import torch.utils.data
import pandas as pd
import os
import torch
import time
import subprocess
import io
import json

import data.loader as loader
import training
import testing
from utils.utils import compute_normalizer
from utils.logger import set_redirects
from utils.utils import save_options
import options.train_options as train_params
from models.model_state import ModelState


def run_main_single(options, path_general, file_name_general):
    start_time = time.time()
    print(time.strftime("%c"))

    if torch.cuda.is_available():
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        gpu_df = pd.read_csv(io.BytesIO(gpu_stats), names=['memory.used', 'memory.free'], skiprows=1)
        print('GPU usage:\n{}'.format(gpu_df))
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
        idx = gpu_df['memory.free'].astype(int).idxmax()
        print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
        torch.cuda.set_device(idx)
        if int(gpu_df.iloc[idx]['memory.free']) < 1000:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            print(f"Using GPU {idx}: {torch.cuda.get_device_name(idx)}")
    else:
        device = torch.device('cpu')
    print('Device: {}'.format(device))

    options['device'] = device
    options['dataset_options'] = train_params.get_dataset_options(options['dataset'])
    options['train_options'] = train_params.get_train_options(options['dataset'])
    options['system_options'] = train_params.get_system_options(options['dataset'], options['dataset_options'], options['train_options'])
    options['model_options'] = train_params.get_model_options(options['model'], options['dataset'], options['dataset_options'])
    options['test_options'] = train_params.get_test_options()

    print('\n\tModel Type: {}'.format(options['model']))
    print('\tDynamic System: {}\n'.format(options['dataset']))

    file_name_general = file_name_general + '_h{}_z{}_n{}'.format(
        options['model_options'].h_dim, options['model_options'].z_dim,
        options['model_options'].n_layers)

    if "mpnt_wt" in options["model_options"]:
        file_name_general = file_name_general + '_mpw' + str(int(options["model_options"].mpnt_wt * 10))

    if "A_prt_idx" in options["dataset_options"]:
        file_name_general = file_name_general + '_A' + str(options["dataset_options"].A_prt_idx)

    if "B_prt_idx" in options["dataset_options"]:
        file_name_general = file_name_general + '_B' + str(options["dataset_options"].B_prt_idx)

    if "C_prt_idx" in options["dataset_options"]:
        file_name_general = file_name_general + '_C' + str(options["dataset_options"].C_prt_idx)

    path = path_general + 'data/'
    if not os.path.exists(path):
        os.makedirs(path)
    set_redirects(path, file_name_general)

    save_options(options, path_general, 'options.txt')

    train_rounds = options["train_rounds"]
    start_from_round = options["start_from"]
    print('Total number of rounds: {}'.format(train_rounds))

    all_vaf = torch.zeros([train_rounds])
    all_rmse = torch.zeros([train_rounds])
    all_nrmse = torch.zeros([train_rounds])
    all_likelihood = torch.zeros([train_rounds])
    all_df = {}

    for i in range(start_from_round, train_rounds):
        file_name_general_i = file_name_general + "_No" + str(i)
        torch.cuda.empty_cache()

        print(' {}/{} round starts'.format(i + 1, train_rounds))
        loaders = loader.load_dataset(
            dataset=options["dataset"],
            dataset_options=options["dataset_options"],
            train_batch_size=options["train_options"].batch_size,
            test_batch_size=options["test_options"].batch_size,
            known_parameter=options["known_parameter"],
            k_max_train=options["dataset_options"].k_max_train,
            k_max_test=options["dataset_options"].k_max_test,
            k_max_val=options["dataset_options"].k_max_val,
            train_rounds=train_rounds,
            ith_round=i)

        if options["normalize"]:
            normalizer_input, normalizer_output = compute_normalizer(loaders['train'])
        else:
            normalizer_input = normalizer_output = None

        modelstate = ModelState(
            seed=options["seed"] + i * 10,
            nu=loaders["train"].nu, ny=loaders["train"].ny,
            model=options["model"],
            options=options,
            normalizer_input=normalizer_input,
            normalizer_output=normalizer_output)
        modelstate.model.to(options['device'])

        if options['do_train']:
            try:
                new_df = training.run_train(
                    modelstate=modelstate,
                    loader_train=loaders['train'],
                    loader_valid=loaders['valid'],
                    options=options,
                    dataframe=[],
                    path_general=path_general,
                    file_name_general=file_name_general_i)
                with open(path_general + file_name_general_i + '_trainingrecord.json', 'w') as f:
                    json.dump(new_df, f)
            except Exception as e:
                print(f"Training failed for round {i + 1}, skipping.")
                print("Error:", e)
                continue

        if options['do_test']:
            try:
                df = pd.DataFrame({})
                df_single = testing.run_test(options, loaders, df, path_general, file_name_general_i)
                df_single = pd.DataFrame(df_single)
                df = pd.concat([df, df_single])

                if options['savelog']:
                    df.to_csv(path + file_name_general_i, index=False)

                all_df[i] = df
                all_vaf[i] = df['vaf'][0]
                all_rmse[i] = df['rmse'][0]
                all_nrmse[i] = df['nrmse'][0]
            except Exception as e:
                print(f"Testing failed for round {i + 1}, skipping.")
                print("Error:", e)

    if options['savelog'] and options['do_test']:
        if not os.path.exists(path):
            os.makedirs(path)
        all_df_list = [i_df for _, i_df in all_df.items()]
        all_df = pd.concat(all_df_list)
        print(all_df)

        file_name = file_name_general + '_multitrain.csv'
        all_df.to_csv(path_general + file_name)
        torch.save(all_vaf, path_general + 'data/' + 'all_vaf.pt')
        torch.save(all_rmse, path_general + 'data/' + 'all_rmse.pt')
        torch.save(all_likelihood, path_general + 'data/' + 'all_likelihood.pt')

    time_el = time.time() - start_time
    hours = time_el // 3600
    minutes = time_el // 60 - hours * 60
    sec = time_el - minutes * 60 - hours * 3600
    print('Total time: {}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, minutes, sec))
    print(time.strftime("%c"))


if __name__ == "__main__":
    options = {}
    main_params_parser = train_params.get_main_options()
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
    options['savelog'] = main_params_parser.savelog
    options['saveoutput'] = main_params_parser.saveoutput
    options['known_parameter'] = main_params_parser.known_parameter
    options['train_rounds'] = main_params_parser.train_rounds
    options['start_from'] = main_params_parser.start_from

    path_general = os.getcwd() + '/log/{}/{}/{}_{}/'.format(
        options['logdir'], options['dataset'],
        options['model'], options['known_parameter'])

    file_name_general = options['dataset']

    run_main_single(options, path_general, file_name_general)
