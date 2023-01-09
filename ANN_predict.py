import numpy as np
import pandas as pd
import torch
from os.path import join
from collections import OrderedDict
import pdb

def predict_Nglyco(location, enzyme_levels):
    location = location.casefold()
    # Attempting to correct the user's input, if needed
    if 'asn' in location and location[3] not in '_0123456789': # User passed something like Asn-24
        location = 'asn_' + location[4:]
    elif 'asn' in location and location[3] in '0123456789': # User passed something like Asn24
        location = 'asn_' + location[3:]
    elif 'fc' in location and location[2] not in {'_', 'd', 'e'}: # User passed something like Fc-DAO
        location = 'fc_' + location[3:]
    elif 'fc' in location and location[2] in {'d', 'e'}: # User passed something like FcDAO
        location = 'fc_' + location[2:]
    elif 'nn' in location:
        location = 'nn_modelnsd'

    # Setup
    hyperparam_dict = {'Asn_24_GnGnF': 'relu', 'Asn_24_GnGnGnF': 'tanhshrink', 'Asn_24_GnGnGnGnF': 'tanhshrink', 'Asn_24_MGnF': 'relu',
                    'Asn_38_GnGnF': 'tanh', 'Asn_38_GnGnGnF': 'tanh', 'Asn_38_GnGnGnGnF': 'relu', 'Asn_38_NaGnGnGnF': 'relu',
                    'Asn_83_GnGnF': 'tanh', 'Asn_83_GnGnGnF': 'relu', 'Asn_83_GnGnGnGnF': 'relu', 'Asn_83_NaGnGnGnF': 'tanhshrink',
                    'Asn_110_Man5': 'relu', 'Asn_110_Man6': 'tanhshrink', 'Asn_110_Man7': 'relu',
                    'Asn_168_GnGnF': 'relu', 'Asn_168_GnGnGnF': 'tanh', 'Asn_168_GnGnGnGnF': 'tanh', 'Asn_168_MGnF': 'tanh',
                    'Asn_538_GnGn': 'relu', 'Asn_538_GnGnF': 'relu', 'Asn_538_MGn': 'relu', 'Asn_538_MGnF': 'tanhshrink',
                    'Asn_745_GnGn': 'tanhshrink', 'Asn_745_GnGnF': 'tanhshrink', 'Asn_745_MGn': 'tanhshrink', 'Asn_745_MGnF': 'tanh',
                    'Fc_DAO_GnGnF': 'relu', 'Fc_EPO_GnGnF': 'relu',
                    'NN_modelNSD_G0F-GnGnF': 'tanhshrink', 'NN_modelNSD_G1F-AGnF': 'tanh', 'NN_modelNSD_G2F-AAF': 'tanh'} 
    if 'asn' in location or 'fc_' in location:
        X_data = pd.read_csv(join('datasets', 'Training-X.csv'), index_col = 0).values
    else:
        X_data = pd.read_csv(join('datasets', 'NN_modelNSD_training-X.csv'), index_col = 0).values
    X_mean = X_data.mean()
    X_std = X_data.std()
    normalized_enzyme = torch.Tensor((enzyme_levels - X_mean)/X_std).cuda() # The ANNs were trained on data with mu = 0 and sigma = 1

    # Making predictions
    for glycan_name, hyperparams in hyperparam_dict.items():
        if location in glycan_name.casefold():
            mydict = torch.load(join('ANN_models', f'ANN_{glycan_name}_dict.pt'))
            # Getting the size of the model from mydict
            layers = []
            for array_name, array in mydict.items():
                if 'weight' in array_name:
                    layers.append(tuple(array.T.shape))
            # Building the model and making predictions
            model = SequenceMLP(layers, hyperparams)
            model.load_state_dict(mydict)
            model.cuda()
            model.eval()
            pred = model(normalized_enzyme).cpu().detach().squeeze()
            print(f'{glycan_name:20}: {pred:.3f}')

class SequenceMLP(torch.nn.Module):
    def __init__(self, layers, activ_fun = 'relu'):
        super(SequenceMLP, self).__init__()
        # Setup to convert string to activation function
        if activ_fun == 'relu':
            torch_activ_fun = torch.nn.ReLU()
        elif activ_fun == 'tanh':
            torch_activ_fun = torch.nn.Tanh()
        elif activ_fun == 'sigmoid':
            torch_activ_fun = torch.nn.Sigmoid()
        elif activ_fun == 'tanhshrink':
            torch_activ_fun = torch.nn.Tanhshrink()
        elif activ_fun == 'selu':
            torch_activ_fun = torch.nn.SELU()
        else:
            raise ValueError(f'Invalid activ_fun. You passed {activ_fun}')
        # Transforming layers list into OrderedDict with layers + activation
        mylist = list()
        for idx, elem in enumerate(layers):
            mylist.append((f'Linear{idx}', torch.nn.Linear(layers[idx][0], layers[idx][1]) ))
            if idx < len(layers)-1:
                mylist.append((f'{activ_fun}{idx}', torch_activ_fun))
        # OrderedDict into NN
        self.model = torch.nn.Sequential(OrderedDict(mylist))

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Loads a trained ANN model and predicts N-linked glycan distribution based on normalized B4GALT1--B4GALT4 levels.')
    parser.add_argument('location', type = str, nargs = 1, help = 'The N-glycosylation location to be predicted. Must be in {Asn_XX, Asn_XXX, Fc_DAO, Fc_EPO, or NN_modelNSD}.')
    parser.add_argument('enzyme_levels', metavar='1 1 1 1', type = float, nargs='+', help='The levels of B4GALT1--B4GALT4, normalized to WT levels.')
    args = parser.parse_args()
    location = args.location[0] # [0] to convert from list to string
    enzyme_levels = args.enzyme_levels
    predict_Nglyco(location, enzyme_levels)

