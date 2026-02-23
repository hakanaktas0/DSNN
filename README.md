# Sheaves Reloaded: A Directional Awakening

<a href="https://arxiv.org/abs/2506.02842"><img src="https://img.shields.io/badge/arXiv-2506.02842-%23B31B1B"></a>


This is the official implementation of the paper "Sheaves Reloaded: A Directional Awakening"

by [Stefano Fiorini](https://scholar.google.com/citations?user=2O-BN9YAAAAJ&hl=it)<sup>+</sup>, [Hakan Aktas](https://www.cst.cam.ac.uk/people/hea39)<sup>+</sup>, [Iulia Duta](https://iuliaduta.github.io/), [Pietro Morerio](https://scholar.google.com/citations?user=lPV9rbkAAAAJ&hl=it),
[Alessio Del Bue](https://scholar.google.co.uk/citations?user=LUzvbGIAAAAJ&hl=en), [Pietro Lio](https://www.cl.cam.ac.uk/~pl219/), [Stefano Coniglio](https://scholar.google.com/citations?user=F9mrD0gAAAAJ&hl=en)

+: Equal Contribution <br>


## Installation

You can install the environment using the YAML file. We used `CUDA 12.1` for this project.
```
conda env create -f environment.yml 
```

## Running the experiments


To run the experiments without a Weights & Biases (wandb) account, first disable `wandb` by running `wandb disabled`. 
Then, for instance, to run the training procedure on `texas`, simply run the example script provided:
```commandline
sh ./exp/scripts/run_texas.sh
```

To run the experiments using `wandb`, first create a [Weights & Biases account](https://wandb.ai/site). Then run the following
commands to log in and follow the displayed instructions:
```bash
wandb online
wandb login
```
Then, you can run the example training procedure on `texas` via:
```bash
export ENTITY=<WANDB_ACCOUNT_ID>
sh ./exp/scripts/run_texas.sh
```
Scripts for the other heterophilic datasets are also provided in `exp/scripts`. The link prediction benchmarks can be run similarly by using the scripts that has 'link'.

### Hyperparameter Sweeps

To run a hyperparameter sweep, you will need a `wandb` account. Once you have an account, you can run an example
sweep as follows:
```bash
export ENTITY=<WANDB_ACCOUNT_ID>
wandb sweep --project sheaf_link config/orth_webkb_sweep.yml
```
This will set up the sweep for a discrete bundle model on the WebKB datasets 
as described in the yaml config at `config/orth_webkb_sweep.yml`.

To run the sweep on a single GPU, simply run the command displayed on screen after running the sweep command above. 
If you like to run the sweep on multiple GPUs, then run the following command by typing in the `SWEEP_ID` received above.
```bash
sh run_sweeps.sh <SWEEP_ID>
```
### Datasets

The WebKB (`texas`, `wisconsin`, `cornell`) and `film` datasets are downloaded on the fly. The rest of the datasets are included in the repository. 

### Synthetic Dataset

The synthetic datasets used in the paper are included under directory `synthetic_dataset`. You can run the experiment similar to others using 

```bash
export ENTITY=<WANDB_ACCOUNT_ID>
sh ./exp/scripts/run_synthetic.sh
```

To change the alpha and beta values, you can change the dataset line in the script to any of the included ones in the `synthetic_dataset` folder.

## Citation

If you find our work helpful, please cite our work as:
```
@article{fiorini2025sheaves,
  title={Sheaves Reloaded: A Directional Awakening},
  author={Fiorini, Stefano and Aktas, Hakan and Duta, Iulia and Coniglio, Stefano and Morerio, Pietro and Del Bue, Alessio and Li{\`o}, Pietro},
  journal={arXiv preprint arXiv:2506.02842},
  year={2025}
}
```