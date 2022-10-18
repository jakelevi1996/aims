# Course 1 - Data, Estimation and Inference

The website for Michael Osborne's DEI lectures can be found [here](https://www.robots.ox.ac.uk/~mosb/aims_cdt/). The website for Tim Rudner's DEI lectures can be found [here](https://tgjr-research.notion.site/Data-Estimation-and-Inference-2022-GPs-c6e81b6fc2ec47f79140c42862d1cadd). This README describes solutions to the [coursework assigment](https://www.robots.ox.ac.uk/~mosb/teaching/AIMS_CDT/CDT_estimation_inference_lab.pdf) for this course, on the subject of Gaussian processes.

## Contents

- [Course 1 - Data, Estimation and Inference](#course-1---data-estimation-and-inference)
  - [Contents](#contents)
  - [Loading and plotting data](#loading-and-plotting-data)
  - [Plotting samples from the GP prior](#plotting-samples-from-the-gp-prior)
  - [Plotting the GP predictive mean and standard deviation](#plotting-the-gp-predictive-mean-and-standard-deviation)
  - [Plotting samples from the GP predictive distribution](#plotting-samples-from-the-gp-predictive-distribution)
  - [Calculating RMSE, log marginal likelihood, and log predictive likelihood](#calculating-rmse-log-marginal-likelihood-and-log-predictive-likelihood)
  - [Optimising hyperparameters](#optimising-hyperparameters)
  - [Epistemic and aleatoric uncertainty](#epistemic-and-aleatoric-uncertainty)
  - [Periodic kernels](#periodic-kernels)
  - [Sum and product kernels](#sum-and-product-kernels)
  - [Sequential prediction](#sequential-prediction)
  - [Predicting gradients](#predicting-gradients)

## Loading and plotting data

Sotonmet (the dataset used in this assignment) can be loaded and plotted using the command `python scripts/course_1_dei/plot_data.py`. This produces the following figure:

![](https://raw.githubusercontent.com/jakelevi1996/aims/main/scripts/course_1_dei/Results/Protected/Sotonmet_data.png)

This script also plots the independent GP predictions provided in `sotonmet.txt`, shown below:

![](https://raw.githubusercontent.com/jakelevi1996/aims/main/scripts/course_1_dei/Results/Protected/Data_and_independent_GP_predictions.png)

## Plotting samples from the GP prior

Samples from the prior of a Gaussian Process can be plotted using the command `python scripts/course_1_dei/plot_prior_samples.py`. Samples from 2 different Gaussian processes (each with a squared exponential kernel) are shown below:

![](https://github.com/jakelevi1996/aims/blob/main/scripts/course_1_dei/Results/Protected/Samples_from_GP_prior,_GP___GaussianProcess(prior_mean_func_Constant(offset_3),_kernel_func_SquaredExponential(length_scale_0.1,_kernel_scale_1),_noise_std_0.001).png?raw=true)

![](https://github.com/jakelevi1996/aims/blob/main/scripts/course_1_dei/Results/Protected/Samples_from_GP_prior,_GP___GaussianProcess(prior_mean_func_Constant(offset_3),_kernel_func_SquaredExponential(length_scale_0.3,_kernel_scale_10),_noise_std_1.0).png?raw=true)

## Plotting the GP predictive mean and standard deviation

## Plotting samples from the GP predictive distribution

## Calculating RMSE, log marginal likelihood, and log predictive likelihood

## Optimising hyperparameters

## Epistemic and aleatoric uncertainty

## Periodic kernels

## Sum and product kernels

## Sequential prediction

## Predicting gradients
