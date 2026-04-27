import torch
import torch.nn as nn
from . import AE_RNN, AE_RNN_XU, AE_RNN_U, AE_RNN_U_SGM, MLP_U, MLP_U_SGM, Liu_U


class DynamicModel(nn.Module):
    def __init__(self, model, num_inputs, num_outputs, options, normalizer_input=None, normalizer_output=None,
                 *args, **kwargs):
        super(DynamicModel, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.normalizer_input = normalizer_input
        self.normalizer_output = normalizer_output
        self.model_name = model
        self.zero_initial_state = False

        model_options = options['model_options']
        system_options = options.get('system_options', {})

        print("model is: ", model)
        if model == 'AE-RNN':
            self.m = AE_RNN(model_options, options['device'], system_options, options['dataset'])
        elif model == 'AE-RNN-U':
            self.m = AE_RNN_U(model_options, options['device'], system_options, options['dataset'])
        elif model == 'AE-RNN-U-SGM':
            self.m = AE_RNN_U_SGM(model_options, options['device'], system_options, options['dataset'])
        elif model == 'AE-RNN-XU':
            self.m = AE_RNN_XU(model_options, options['device'], system_options, options['dataset'])
        elif model == 'MLP-U':
            self.m = MLP_U(model_options, options['device'], system_options, options['dataset'])
        elif model == 'MLP-U-SGM':
            self.m = MLP_U_SGM(model_options, options['device'], system_options, options['dataset'])
        elif model == 'LIU-U':
            self.m = Liu_U(model_options, options['device'], system_options, options['dataset'])
        else:
            raise Exception("Unimplemented model: {}".format(model))

    @property
    def num_model_inputs(self):
        return self.num_inputs + self.num_outputs if self.ar else self.num_inputs

    def forward(self, u, y=None):
        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)
        if y is not None and self.normalizer_output is not None:
            y = self.normalizer_output.normalize(y)
        return self.m(u, y, self.normalizer_input, self.normalizer_output)

    def generate(self, u, y=None):
        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)
        if y is not None and self.normalizer_output is not None:
            y = self.normalizer_output.normalize(y)

        y_sample, y_sample_mu, y_sample_sigma, z = self.m.generate(u, self.normalizer_input, self.normalizer_output)

        if self.normalizer_output is not None:
            y_sample = self.normalizer_output.unnormalize(y_sample)
            y_sample_mu = self.normalizer_output.unnormalize_mean(y_sample_mu)
            y_sample_sigma = self.normalizer_output.unnormalize_sigma(y_sample_sigma)

        return y_sample, y_sample_mu, y_sample_sigma, z
