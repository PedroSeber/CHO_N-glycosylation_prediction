import torch
import shap
import numpy as np
from pandas import read_csv
from ANN_predict import hyperparam_dict, SequenceMLP
from os.path import join as os_join # join is too generic of a name to be imported directly
import matplotlib.pyplot as plt

def shap_analysis(exp_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # NOTE: to make shap work in a GPU, I had to edit two lines in /shap/explainers/_deep/deep_pytorch.py . These changes are not recommended if using layers that behave differently in train vs. eval (such as dropout)
    # Attempting to correct the user's input
    exp_name = exp_name[:3].casefold() + exp_name[3:]
    if 'asn' in exp_name and exp_name[3] not in '0123456789': # User passed something like Asn-24 or Asn_24 (the latter is correct, but we need to undo the casefold for this function)
        exp_name = 'Asn_' + exp_name[4:]
    elif 'asn' in exp_name: # User passed something like Asn24
        exp_name = 'Asn_' + exp_name[3:]
    elif 'fc' in exp_name and exp_name[2] not in {'d', 'e'}: # User passed something like Fc-DAO or Fc_DAO (the latter is correct, but we need to undo the casefold for this function)
        exp_name = 'Fc_' + exp_name[3:]
    elif 'fc' in exp_name: # User passed something like FcDAO
        exp_name = 'Fc_' + exp_name[2:]
    train_data_X = torch.Tensor(read_csv(os_join('datasets', 'Training-X.csv'), index_col = 0).values)
    test_data_X = torch.Tensor(read_csv(os_join('datasets', 'Test-X.csv'), index_col = 0).values)
    y_name = '_'.join(exp_name.split('_')[:-1])
    train_data_y = read_csv(os_join('datasets', f'{y_name}_training-y.csv'), index_col = 0).loc[:, exp_name.split('_')[-1]].values
    train_data_y = np.array([train_data_y[0], train_data_y[1:3].mean(), train_data_y[3:6].mean(), train_data_y[6:9].mean(), train_data_y[9:].mean()])
    test_data_y = np.atleast_1d(np.round(read_csv(os_join('datasets', f'{y_name}_test-y.csv'), index_col = 0).loc[:, exp_name.split('_')[-1]].values.mean(), 4))
    # Model preparation
    mydict = torch.load(os_join('ANN_models', f'ANN_{exp_name}_dict.pt'), map_location = torch.device(device))
    # Getting the size of the model from mydict
    layers = []
    for array_name, array in mydict.items():
        if 'weight' in array_name:
            layers.append(tuple(array.T.shape))
    # Building the model and making predictions
    if hyperparam_dict[exp_name] != 'tanhshrink':
        model = SequenceMLP(layers, hyperparam_dict[exp_name])
    else: # Shap does not support tanhshrink. We are approximating it with tanh
        print('Approximating tanhshrink with tanh')
        model = SequenceMLP(layers, 'tanh')
    model.load_state_dict(mydict)
    model.to(device)
    model.eval()
    # Building the shap explainer + shap explanations (training data)
    bg = torch.Tensor([[0, 0, 0, 0]])
    fg_orig = torch.Tensor([[1, 1, 1, 1],
                       [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
                       [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1],
                       [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Normalizing the enzyme levels data
    X_mean = train_data_X.mean()
    X_std = train_data_X.std()
    bg = (bg - X_mean)/X_std
    fg = (fg_orig - X_mean)/X_std
    explainer = shap.DeepExplainer(model, bg.to(device))
    shap_values = np.array(explainer.shap_values(fg.to(device), check_additivity = False))
    print(f'{np.round( (shap_values*fg.numpy()).sum(axis=1), 4)[[0, -4, -3, -2, -1]] + test_data_y} vs.\n{np.round(train_data_y, 4)} and {test_data_y}')
    # Heatmap of the shap values per experiment and enzyme levels
    fig, ax = plt.subplots(figsize = (4, 7), dpi = 200)
    txt_size = 10
    #ax.tick_params(top = True, labeltop = True, right = True, labelright = True) # Also include position tick labels on the top of the heatmap, and AA tick labels on the right of the heatmap
    ax.tick_params(top = True, labeltop = True) # Also include position tick labels on the top of the heatmap, and AA tick labels on the right of the heatmap
    im = ax.imshow(shap_values)
    ax.set_xticks(range(0, 4), labels = ['B1', 'B2', 'B3', 'B4'])
    ax.set_yticks(range(shap_values.shape[0]), labels = [''.join(elem) for elem in fg_orig.int().numpy().astype(str)])
    cbar = ax.figure.colorbar(im, ticks = [])
    for idx_X in range(shap_values.shape[1]):
        for idx_Y in range(shap_values.shape[0]):
            if shap_values[idx_Y, idx_X] < 0:
                color = '#FFFF00' # Yellow
            else:
                color = '#0000FF' # Blue
            if np.abs(shap_values[idx_Y, idx_X])*100 >= 5e-2:
                text = ax.text(idx_X, idx_Y, f'{shap_values[idx_Y, idx_X]*100:.1f}', ha = "center", va = "center", color = color, size = txt_size)
    fig.tight_layout()
    plt.savefig(f'Shap_values_{exp_name}.svg')

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Loads a trained RNN model and performs Shapley additive explanations for interpretability.')
    parser.add_argument('exp_name', type = str, nargs = 1, help = 'The location and name of a glycan to be analyzed (e.g.: Asn_24_GnGnF).')
    args = parser.parse_args()
    exp_name = args.exp_name[0] # [0] to convert from list to string
    shap_analysis(exp_name)

