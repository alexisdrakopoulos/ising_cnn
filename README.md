# CNN project for Ising model generated via MCMC

This project was part of my BSc dissertation at the University of Strathclyde and continued over summer 2019 as a research project.

The project has two major components, the MCMC system generation and the actual CNN training, prediction and evaluation. This project was started with the goal of producing a paper on an overview of deep regression in late 2018/early 2019, however another paper came up [A Comprehensive Analysis of Deep Regression](https://ieeexplore.ieee.org/abstract/document/8686063/) which was not only far more expansive in its analysis but also had access to significantly more compute. It was however good to see that the same conclusions were drawn.

This README is split into an introduction, which you are reading now, a config/dependency section, a MCMC module section, the CNN section, then a theory section. The Theory section is not required to run this.

Please keep in mind this project was made in mind with utilizing AWS infrastructure, particularly S3 for file storage/backups and EC2 for compute. the EC2 p3 instances were used for training using the AWS DLAMI Version 24.

## How to Run Basics

Nothing in `code` directory needs to be directly touched/changed. Everything that you should need can be called from two scripts, `generate_systems.py`  and `main_gpu.py` . The former generates systems using MCMC, and can prepare synthetic Ising systems with the latter being used for training. Inference and evaluation of the CNN models does need access to `code/run_predictions.py`.

Examples of how to call these would be:

`python3 generate_system.py mcmc 5000 `

`python3 generate_systems.py combine`

`python3 generate_systems.py compress`

`python3 main_gpu.py`

`cd code`

`python3 run_predictions.py`

This would generate 5000 systems (and 10,000 .npy files), combine them, then compress them into a nice npz format, then train your model (assuming `models.csv` is filled), cd into `code` folder, then predict using that model.

## Configuration, Dependencies and Directory Structure

This section discusses the `config.ini` file, as well as the folder structure that the project comes with and its behaviors in terms of filling the folder structure. Dependencies are also discussed.

### Config File

All module paths are configured using the `config.ini` file, as well as calling AWS S3.

### Dependencies

This project uses a few libraries, the obvious ones are NumPy and Numba. Pathlib is used for all file directory management and tqdm is used for progress bars.

The deep learning module was built through Keras API in TensorFlow 1.13, it will NOT work for TensorFlow 2.0, but can quite easily be upgraded to it. Sci-kit learn is also used purely for its accuracy metric. As noted previously, AWS DLAMI (Ubuntu Version 24) was used for running all training/inference code. MCMC modules were trained on AWS EC2 C4/C5 instances using [Clear Linux](https://clearlinux.org/) due to its nice Intel Xeon optimizations that allowed for 20% better performance.

### Directory Structure

```
​```
> code
	> __init__.py
	> config.ini
	> logging_module.py
	> metropolis_ising.py
	> model.py
	> model_details.csv
	> model_evaluations.csv
	> predictions.py
	> run_predictions.py
	> utils.py
> data
	> ising
	> logs
		> loss_histories
		> model_logs
		> training_logs
	> models
	> predictions
	models.csv
generate_systems.py
main_gpu.py
README.md
But let's throw in a <b>tag</b>.
​```
```

## MCMC Module

The MCMC module is really the Metropolis-Hastings module. The core implementation can be found in `code/metropolis_ising.py` where the function `mcmc_ising` is a Metropolis-Hastings implementation of the 2D Isotropic Ising model with periodic boundary conditions.

You should not really need to call this function directly, 

## CNN Module

## Theory, Questions that were asked/Answers
