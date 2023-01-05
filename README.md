## Linear and Neural Network Models for Predicting N-glycosylation in Chinese Hamster Ovary Cells Based on B4GALT Levels
These are the datasets and model files associated with the publication [Linear and Neural Network Models for Predicting N-glycosylation in Chinese Hamster Ovary Cells Based on B4GALT Levels](TODO). This work uses linear models and ANNs to predict the distribution of glycans on potential N-glycosylation sites. The models were trained on data containing normalized CHO cell B4GALT levels.<br>

### Reproducing the models and plots
Download the [datasets](datasets) folder and run the [SPA\_glycosylation\_model.py](SPA_glycosylation_model.py) file without any flags (`python SPA_glycosylation_model.py`) to recreate the cross-validation results, and run with the `--nested` flag (`python SPA_glycosylation_model.py --nested`) to recreate the nested validation results. To recreate the ANN results, download the [ANN\_train.ipynb](ANN_train.ipynb) file, change the first cell as needed, and run the notebook.<br>
To recreate the plots, download the [result\_csv\_files](result_csv_files) and [result\_csv\_files\_nested](result_csv_files_nested) folders, then run the [make\_results\_plots.py](make_results_plots.py) file. Most plots will be generated in the first folder, but the nested validation PRE distribution will be generated in the nested results folder.

### Using the models to predict glycan distributions
The Conda environment defining the specific packages and version numbers used in this work is available as [ANN\_environment.yaml](ANN_environment.yaml). To use our trained models, run the [ANN\_predict.py](ANN_predict.py) file.
