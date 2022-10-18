# Course 1 - Data, Estimation and Inference

The website for Michael Osborne's DEI lectures can be found [here](https://www.robots.ox.ac.uk/~mosb/aims_cdt/). The website for Tim Rudner's DEI lectures can be found [here](https://tgjr-research.notion.site/Data-Estimation-and-Inference-2022-GPs-c6e81b6fc2ec47f79140c42862d1cadd). This README describes solutions to the [coursework assigment](https://www.robots.ox.ac.uk/~mosb/teaching/AIMS_CDT/CDT_estimation_inference_lab.pdf) for this course, on the subject of Gaussian Processes.

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

![](./Results/Protected/Sotonmet_data.png)

This script also plots the independent GP predictions provided in `sotonmet.txt`, shown below:

![](./Results/Protected/Data_and_independent_GP_predictions.png)

## Plotting samples from the GP prior

Samples from the prior of a Gaussian Process can be plotted using the command `python scripts/course_1_dei/plot_prior_samples.py`. Samples from 2 different Gaussian Processes (each with a squared exponential kernel) are shown below:

![](./Results/Protected/Samples_from_GP_prior,_GP___GaussianProcess_prior_mean_func_Constant_offset_3_,_kernel_func_SquaredExponential_length_scale_0.1,_kernel_scale_1_,_noise_std_0.001_.png)

![](./Results/Protected/Samples_from_GP_prior,_GP___GaussianProcess_prior_mean_func_Constant_offset_3_,_kernel_func_SquaredExponential_length_scale_0.3,_kernel_scale_10_,_noise_std_1.0_.png)

## Plotting the GP predictive mean and standard deviation

The mean and standard deviation of the predictive distribution of a Gaussian Process can be plotted using the command `python scripts/course_1_dei/plot_gp_predictions.py`. The predictive distributions of 2 different Gaussian Processes (with identical parameters to the 2 Gaussian Processes used for plotting samples from the prior distribution above) are shown below. Note that, although the first Gaussian Process produces a *prior* distribution which looks like a more plausible explanation for the training data, the second Gaussian Process produces a *predictive* distribution which looks like a much better fit to the training data.

![](./Results/Protected/Data_and_GP_predictions,_GP___GaussianProcess_prior_mean_func_Constant_offset_3_,_kernel_func_SquaredExponential_length_scale_0.1,_kernel_scale_1_,_noise_std_0.001_.png)

![](./Results/Protected/Data_and_GP_predictions,_GP___GaussianProcess_prior_mean_func_Constant_offset_3_,_kernel_func_SquaredExponential_length_scale_0.3,_kernel_scale_10_,_noise_std_1.0_.png)

## Plotting samples from the GP predictive distribution

Joint samples from the predictive distribution of a Gaussian process can be plotted using the command `python scripts/course_1_dei/plot_predictive_samples.py`. Joint samples from the predictive distributions of the same 2 Gaussian Processes are shown below. Note that, although the first Gaussian Process produces a *prior* distribution which looks like a plausible explanation for the training data, and the second Gaussian Process produces a *predictive* distribution which looks like a good fit to the training data, neither Gaussian Process produces a joint predictive distribution whose *samples* look like a plausible explanation for the training data.

![](./Results/Protected/Data_and_GP_predictions_and_predictive_samples,_GP___GaussianProcess_prior_mean_func_Constant_offset_3_,_kernel_func_SquaredExponential_length_scale_0.1,_kernel_scale_..._.png)

![](./Results/Protected/Data_and_GP_predictions_and_predictive_samples,_GP___GaussianProcess_prior_mean_func_Constant_offset_3_,_kernel_func_SquaredExponential_length_scale_0.3,_kernel_scale_..._.png)

## Calculating RMSE, log marginal likelihood, and log predictive likelihood

## Optimising hyperparameters

## Epistemic and aleatoric uncertainty

## Periodic kernels

## Sum and product kernels

## Sequential prediction

## Predicting gradients
