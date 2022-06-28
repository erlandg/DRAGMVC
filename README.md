# DRAGMVC

## Specs (tested on)

- Python version: 3.7  
- PyTorch version: 1.10.0  
- CUDA version: 11.3  
- PyTorch Geometric version: 2.0.4  


## Dataset format

Datasets (of .npz format) contain:  
```
n_views: The number of views, V
labels: Integer labels. Shape (n,)  
graph: Dense affinity matrix. Shape (n, n)
view_0: Data for first view. Shape (n, ...)  
  .  
  .  
  .  
view_V: Data for view V. Shape (n, ...)  
```
and are placed in ./data/processed/  

For more information about the MIMIC dataset and how it may be constructed into a graph multi-view dataset, see https://github.com/erlandg/create-mimic-dataset.

## Training the model

The model is trained from the ./src/ directory by
```bash
python -m models.train -c <CONFIG_NAME> [--logs] [--<CONFIG_PARAMETER>=<PARAMETER_VALUE>]
```
where `<CONFIG_NAME>` is the name of a config in ./src/config/experiments/, e.g. *gmvc_contrast*. The config file (in this case *./src/config/experiments/graph_mvc.py*) may be altered directly or recreated to comply with different datasets or different configurations.  

The `--logs` argument is used to enable [Weights & Biases](https://wandb.ai) logging. For this to work, two values in *./src/models/train.py* should be altered. Firstly, we define `entity = WANDB_USER` in the file's `main` function; Secondly, in calling `main`, we define the project name of the W&B project, e.g. `main("wandb-project")`.  

Lastly, to allow for W&B sweeps, configs may be added as callable arguments. The config parameters correspond to those of the config file *./src/config/defaults.py*. For unique parameter names we may define, e.g., `epsilon_features=0.20`, while for ambiguous parameters, we must define its position in the configuration hierarchy separated by "__", e.g.: `model_config__graph_attention_configs__activation=relu`.  

## Evaluating the model

The model may be evaluated by running (also from ./src/)
```bash
python -m models.evaluate -c <CONFIG_NAME> -t <MODEL_TAG> [--plot]
```
with a tag corresponding to the model's save path in ./models/. This will extract the best (minimum loss) model and produce fusion plots and sample X-rays (if applicable) and performance metrics.
