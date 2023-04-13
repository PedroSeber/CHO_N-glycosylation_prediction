from numpy import maximum as np_maximum
import pandas as pd
import torch
from os.path import join as osjoin
from collections import OrderedDict
import warnings
import pdb

def predict_Nglyco(location, enzyme_levels):
    """
    A function to predict the distribution of N-glycans in CHO cells based on B4GALT1-B4GALT4 levels.
    Return the predicted N-glycan levels either directly on the terminal (when one experiment is directly passed to enzyme_levels) or as a new .csv (when the path to a .csv is passed to enzyme_levels)

    Parameters
    ----------
    location : string
        The antibody location of the glycosylation site whose N-glycan distribution will be predicted
    enzyme_levels : array or string
        If array, the normalized enzyme levels of B4GALT1-B4GALT4.
        If string, the path to a .csv file with (N+1)x5 data representing the levels of B4GALT1-B4GALT4 in N experiments plus a header and a row index
    """
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
    hyperparam_dict = {
        'Asn_24_AAF': 'tanhshrink', 'Asn_24_GnGnF': 'relu', 'Asn_24_GnGnGnF': 'tanhshrink', 'Asn_24_GnGnGnGnF': 'tanhshrink', 'Asn_24_MGnF': 'relu', 'Asn_24_NaAAF': 'tanhshrink', 'Asn_24_NaAF': 'selu', 'Asn_24_NaGnF': 'tanhshrink', 'Asn_24_NaGnGnF': 'relu', 'Asn_24_NaGnGnGnF': 'relu', 'Asn_24_NaNaAF': 'tanhshrink', 'Asn_24_NaNaF': 'relu', 'Asn_24_NaNaNaAF': 'tanhshrink', 'Asn_24_NaNaNaF': 'tanhshrink',
        'Asn_38_AGnGnGnF': 'tanhshrink', 'Asn_38_GnGnF': 'tanh', 'Asn_38_GnGnGnF': 'tanh', 'Asn_38_GnGnGnGnF': 'relu', 'Asn_38_NaAAF': 'relu', 'Asn_38_NaAF': 'tanhshrink', 'Asn_38_NaAGnGnF': 'selu', 'Asn_38_NaGnGnF': 'tanhshrink', 'Asn_38_NaGnGnGnF': 'relu', 'Asn_38_NaNaAAF': 'tanhshrink', 'Asn_38_NaNaAF': 'tanhshrink', 'Asn_38_NaNaGnGnF': 'tanhshrink', 'Asn_38_NaNaNaAF': 'relu', 'Asn_38_NaNaNaF': 'selu', 'Asn_38_NaNaNaGnF': 'tanhshrink', 'Asn_38_NaNaNaNaF': 'relu',
        'Asn_83_GnGnF': 'tanh', 'Asn_83_GnGnGnF': 'relu', 'Asn_83_GnGnGnGnF': 'relu', 'Asn_83_NaGnGnGnF': 'tanhshrink', 'Asn_83_NaNaAAF': 'tanhshrink', 'Asn_83_NaNaAF': 'tanh', 'Asn_83_NaNaAGnF': 'relu', 'Asn_83_NaNaGnF': 'tanh', 'Asn_83_NaNaNaAF': 'tanhshrink', 'Asn_83_NaNaNaF': 'tanh', 'Asn_83_NaNaNaNaF': 'relu',  'Asn_83_NaNaNaNaF+LacNAc': 'relu',
        'Asn_110_Man5': 'relu', 'Asn_110_Man6': 'tanhshrink', 'Asn_110_Man7': 'relu',
        'Asn_168_AGnF': 'tanhshrink', 'Asn_168_GnGnF': 'relu', 'Asn_168_GnGnGnF': 'tanh', 'Asn_168_GnGnGnGnF': 'tanh', 'Asn_168_MGnF': 'tanh', 'Asn_168_NaAAF': 'tanhshrink', 'Asn_168_NaAF': 'relu', 'Asn_168_NaAGnF': 'relu', 'Asn_168_NaGnF': 'tanhshrink', 'Asn_168_NaGnGnF': 'relu', 'Asn_168_NaNaAF': 'selu', 'Asn_168_NaNaF': 'relu', 'Asn_168_NaNaNaAF': 'selu', 'Asn_168_NaNaNaF': 'selu',
        'Asn_538_AAF': 'tanh', 'Asn_538_AGnF': 'tanh', 'Asn_538_GnGn': 'relu', 'Asn_538_GnGnF': 'relu', 'Asn_538_MGn': 'relu', 'Asn_538_MGnF': 'tanhshrink', 'Asn_538_NaA': 'tanhshrink', 'Asn_538_NaAF': 'relu', 'Asn_538_NaGn': 'relu', 'Asn_538_NaGnF': 'selu', 'Asn_538_NaNa': 'tanhshrink', 'Asn_538_NaNaF': 'selu',
        'Asn_745_AA': 'tanhshrink', 'Asn_745_AAF': 'tanh', 'Asn_745_AGn': 'tanhshrink', 'Asn_745_AGnF': 'selu', 'Asn_745_GnGn': 'tanhshrink', 'Asn_745_GnGnF': 'tanhshrink', 'Asn_745_MGn': 'tanhshrink', 'Asn_745_MGnF': 'tanh', 'Asn_745_NaA': 'selu', 'Asn_745_NaAF': 'tanh', 'Asn_745_NaGn': 'tanhshrink', 'Asn_745_NaGnF': 'tanhshrink', 'Asn_745_NaNa': 'relu', 'Asn_745_NaNaF': 'relu',
        'Fc_DAO_AAF': 'tanh', 'Fc_DAO_AGnF': 'relu', 'Fc_DAO_GnGnF': 'relu', 'Fc_DAO_MGnF': 'tanhshrink',
        'Fc_EPO_AAF': 'tanh', 'Fc_EPO_AGnF': 'tanhshrink', 'Fc_EPO_GnGn': 'tanhshrink', 'Fc_EPO_GnGnF': 'relu', 'Fc_EPO_MGnF': 'relu', 'Fc_EPO_NaAF': 'relu',
        'NN_modelNSD_G0-GnGn': 'tanh', 'NN_modelNSD_G0F-GnGnF': 'tanhshrink', 'NN_modelNSD_G1F-AGnF': 'tanh', 'NN_modelNSD_G2F-AAF': 'tanh'}
    if 'asn' in location or 'fc_' in location:
        X_data = pd.read_csv(osjoin('datasets', 'Training-X.csv'), index_col = 0).values
    else:
        X_data = pd.read_csv(osjoin('datasets', 'NN_modelNSD_training-X.csv'), index_col = 0).values
    X_mean = X_data.mean()
    X_std = X_data.std()
    if len(enzyme_levels) == 1: # User passed multiple experiments as a .csv through the command line, so the inputs are passed as lists
        enzyme_levels = enzyme_levels[0]
    if isinstance(enzyme_levels, str): # Reading the .csv
        original_path = enzyme_levels # For the failsafe below + convenience when saving the results
        enzyme_levels = pd.read_csv(enzyme_levels, index_col = 0)
        temp_pred = []
        temp_glycan_names = []
        if enzyme_levels.shape[1] == 3:
            enzyme_levels = pd.read_csv(original_path)
            warnings.warn('Apparently you did not include row headers in your .csv - that is, your .csv is (N+1)x4 instead of (N+1)x5. Assuming all 4 columns are levels of B4GALT')
    normalized_enzyme = torch.Tensor((enzyme_levels.values - X_mean)/X_std).cuda() # The ANNs were trained on data with mu = 0 and sigma = 1

    # Formatting of the results
    if location == 'asn_83': # Asn_83 has one glycan with a very long name
        spacing = 23
    elif location == 'nn_modelnsd':
        spacing = 21
    else:
        spacing = 17
    # Making predictions
    for glycan_name, hyperparams in hyperparam_dict.items():
        if location in glycan_name.casefold():
            mydict = torch.load(osjoin('ANN_models', f'ANN_{glycan_name}_dict.pt'))
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
            pred = np_maximum( model(normalized_enzyme).cpu().detach().squeeze(), 0 ) # Glycan share cannot be < 0
            if normalized_enzyme.shape[1] > 1:
                temp_pred.append(pred.numpy())
                temp_glycan_names.append( glycan_name.split('_')[-1] )
            else:
                print(f'{glycan_name:{spacing}}: {pred:.3f}')
    # Saving the results as a .csv (if enzyme_levels were a .csv)
    output = pd.DataFrame(temp_pred, index = temp_glycan_names, columns = enzyme_levels.index).T.round(3) # Setting up the DataFrame with the right row/column names + rounding to 3 decimal places
    output.to_csv(''.join(original_path.split('.')[:-1]) + f'_predictions_{location}.csv')

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
    parser.add_argument('enzyme_levels', metavar='1 1 1 1', nargs='+', help='The levels of B4GALT1--B4GALT4, normalized to WT levels. If location == NN_modelNSD, the 7 levels of nucleotide sugars, normalized to 0.5')
    args = parser.parse_args()
    location = args.location[0] # [0] to convert from list to string
    enzyme_levels = args.enzyme_levels
    predict_Nglyco(location, enzyme_levels)

