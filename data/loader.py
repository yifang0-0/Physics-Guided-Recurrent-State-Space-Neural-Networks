from data.base import DataLoaderExt
from data.toy_lgssm_5_pre import create_toy_lgssm_5_datasets
from data.IndustRobo import create_industrobo_datasets
from data.cascTank import create_cascadedtank_datasets


def load_dataset(dataset, dataset_options, train_batch_size, test_batch_size, **kwargs):

    if dataset == 'cascaded_tank':
        dataset_train, dataset_valid, dataset_test = create_cascadedtank_datasets(
            dataset_options.seq_len_train,
            dataset_options.seq_len_val,
            dataset_options.seq_len_test,
            **kwargs)

    elif dataset == 'toy_lgssm_5_pre':
        dataset_train, dataset_valid, dataset_test = create_toy_lgssm_5_datasets(
            dataset_options.seq_len_train,
            dataset_options.seq_len_val,
            dataset_options.seq_len_test,
            **kwargs)

    elif dataset == 'industrobo':
        dataset_train, dataset_valid, dataset_test = create_industrobo_datasets(
            seq_len_train=dataset_options.seq_len_train,
            seq_len_val=dataset_options.seq_len_val,
            seq_len_test=dataset_options.seq_len_test,
            seq_stride=dataset_options.seq_stride,
            sample_rate=dataset_options.dt,
            input_lev=dataset_options.input_channel,
            file_name=dataset_options.if_simulation,
            **kwargs)

    else:
        raise Exception("Dataset not implemented: {}".format(dataset))

    loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
    loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
    loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)

    return {"train": loader_train, "valid": loader_valid, "test": loader_test}
