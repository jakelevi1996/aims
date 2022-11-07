# Machine Learning Assignment

## Contents

- [Machine Learning Assignment](#machine-learning-assignment)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Visualising predictions](#visualising-predictions)
  - [Comparing CPU vs GPU](#comparing-cpu-vs-gpu)
  - [Comparing momentum parameters](#comparing-momentum-parameters)
  - [Comparing number of hidden layers](#comparing-number-of-hidden-layers)
  - [Comparing dimension of hidden layers](#comparing-dimension-of-hidden-layers)
  - [Comparing batch sizes](#comparing-batch-sizes)
  - [Comparing activation functions](#comparing-activation-functions)
  - [Adversarial examples](#adversarial-examples)
  - [Training an RNN on the works of Shakespeare](#training-an-rnn-on-the-works-of-shakespeare)
  - [Comparing LSTM vs RNN on the works of Shakespeare](#comparing-lstm-vs-rnn-on-the-works-of-shakespeare)

## Introduction

The instructions for this assignment can be found [here](https://github.com/gbaydin/ml-aims-mt2022/tree/main/assessed-assignment).

## Visualising predictions

Shown below are predictions for one example of each digit in the MNIST test set, from a MLP trained for 5 epochs on the MNIST training set, with 2 hidden layers, 400 hidden units per layer, Relu activation functions applied to the output of each hidden layer, trained with stochastic gradient descent (SGD) with a momentum parameter of 0.8:

![](./Results/Protected/Test_set_predictions.png)

## Comparing CPU vs GPU

The script `scripts/course_4_ml/assignment/compare_cpu_gpu.py` compares inference with a MLP on MNIST on the CPU and GPU, for 2 different sized models, and runs in about 5 minutes 40.9 seconds. The plots below show the comparison for a MLP with 2 hidden layers and 400 hidden units per layer, and with 4 hidden layers and 800 hidden units per layer, respectively, running on a laptop with an NVIDIA GeForce MX250 GPU. The GPU offers a speed-up for training in both cases, but the speed-up is much more significant for the larger model. Interestingly, there doesn't appear to be as much speed up for test set evaluations, presumably because the GPU speed benefit is offset by copying data from CPU to GPU.

![](./Results/Protected/MNIST_cross_entropy_loss_over_5_epochs_vs_time,_CPU_vs_GPU,_2_hidden_layers,_400_hidden_units.png)

![](./Results/Protected/MNIST_cross_entropy_loss_over_5_epochs_vs_time,_CPU_vs_GPU,_4_hidden_layers,_800_hidden_units.png)

The test set prediction accuracies after each epoch for the 4 training sessions are summarised in the table below:

Number of epochs | Small model, CPU | Large model, CPU | Small model, GPU | Large model, GPU
--- | --- | --- | --- | ---
0 |  7.990% |  8.750% |  7.990% |  8.750%
1 | 73.790% | 82.690% | 73.860% | 82.390%
2 | 82.310% | 87.600% | 82.330% | 87.380%
3 | 85.130% | 89.750% | 85.070% | 89.550%
4 | 86.900% | 90.740% | 86.930% | 90.600%
5 | 88.070% | 91.480% | 88.020% | 91.430%

The time taken for each of the 4 training sessions is summarised in the table below:

Small model, CPU | Large model, CPU | Small model, GPU | Large model, GPU
--- | --- | --- | ---
1 minutes 10.1 seconds | 2 minutes 7.7 seconds | 59.2 seconds | 1 minutes 3.2 seconds

Below are equivalent results when running the same script on a server using an NVIDIA TITAN V GPU:

![](./Results/Protected/server_MNIST_cross_entropy_loss_over_5_epochs_vs_time,_CPU_vs_GPU,_2_hidden_layers,_400_hidden_units.png)

![](./Results/Protected/server_MNIST_cross_entropy_loss_over_5_epochs_vs_time,_CPU_vs_GPU,_4_hidden_layers,_800_hidden_units.png)

## Comparing momentum parameters

The following shows a comparison in the learning curves for different momentum parameters, running on the server using the NVIDIA TITAN V GPU:

![](./Results/Protected/MNIST_cross_entropy_loss_over_5_epochs_vs_time,_comparing_momentum_parameters.png)

## Comparing number of hidden layers

The following shows a comparison in the learning curves for different numbers of hidden layers:

![](./Results/Protected/MNIST_cross_entropy_loss_over_5_epochs_vs_time,_comparing_number_of_hidden_layers.png)

## Comparing dimension of hidden layers

The following shows a comparison in the learning curves for different numbers of hidden units per hidden layer:

![](./Results/Protected/MNIST_cross_entropy_loss_over_5_epochs_vs_time,_comparing_dimension_of_hidden_layers.png)

## Comparing batch sizes

The following shows a comparison in the learning curves for different batch sizes:

![](./Results/Protected/MNIST_cross_entropy_loss_over_5_epochs_vs_time,_comparing_batch_size.png)

## Comparing activation functions

The following shows a comparison in the learning curves for different activation functions:

![](./Results/Protected/MNIST_cross_entropy_loss_over_5_epochs_vs_time,_comparing_hidden_activation_functions.png)

## Adversarial examples

The script `scripts/course_4_ml/assignment/adversarial_example.py` can be used to generate adversarial examples, for example:

![](./Results/Protected/Test_set_predictions_with_adversarial_example.png)

Below is the training curve for the loss function between the prediction and the adversarial target vs iteration:

![](./Results/Protected/Adversarial_loss_vs_iteration,_5000_iterations,_maximum_pixel_perturbation___0.100.png)

## Training an RNN on the works of Shakespeare

The script `scripts/course_4_ml/assignment/compare_rnn_cpu_gpu.py` trains an RNN on the works of Shakespeare, and compares the time taken between CPU and GPU. The results from running this script on a laptop with an NVIDIA GeForce MX250 GPU are shown below:

![](./Results/Protected/Shakespeare_RNN_mean_cross_entropy_loss_vs_time_over_1000_batches_of_64_characters_each,_CPU_vs_GPU.png)

## Comparing LSTM vs RNN on the works of Shakespeare

The script `scripts/course_4_ml/assignment/compare_rnn_lstm.py` trains both a LSTM and a RNN on the works of Shakespeare, and compares the learning curves. The results from running this script on a laptop with an NVIDIA GeForce MX250 GPU are shown below:

![](./Results/Protected/small_lstm_Shakespeare_RNN_vs_LSTM,_mean_cross_entropy_loss_vs_time_over_1000_batches_of_64_characters_each.png)

In the image above, the performance of the LSTM is not as good as the RNN, but the LSTM also has many fewer parameters. The following image shows the results of an experiment which is the same as the previous one, except every MLP in the LSTM has 2 hidden layers instead of 1:

![](./Results/Protected/medium_lstm_Shakespeare_RNN_vs_LSTM,_mean_cross_entropy_loss_vs_time_over_1000_batches_of_64_characters_each.png)

The following image shows the results of an experiment which is the same as the previous one, except every MLP in the LSTM has 2 hidden layers instead of 1 and also three times as many hidden units in every hidden layer, and the hidden and cell states are also three times larger:

![](./Results/Protected/large_Shakespeare_RNN_vs_LSTM,_mean_cross_entropy_loss_vs_time_over_1000_batches_of_64_characters_each.png)

The following image shows the results of an experiment which is the same as the previous one, except every MLP in the LSTM has only 1 hidden layer but also three times as many hidden units in every hidden layer and in the hidden and cell states than in the original experiment:

![](./Results/Protected/wide_shallow_Shakespeare_RNN_vs_LSTM,_mean_cross_entropy_loss_vs_time_over_1000_batches_of_64_characters_each.png)

All of the above results are training on strings which are 64 characters long, but using only a single string for every training step. The following figure shows the results from an experiment which is the same as the previous one, except that a batch of 100 different strings (each 64 characters long) is used for every training step. The figure shows that the models learn more slowly (although only about 4 times more slowly, despite using 100 times as much data on every training step), but the resulting learning curves are much less noisy:

![](./Results/Protected/Shakespeare_RNN_vs_LSTM,_mean_cross_entropy_loss_vs_time_over_1000_batches_of_100_substrings_with_64_characters_each.png)

Below are training curves from some longer (30 minute) training runs for both the LSTM and RNN, showing slow but steady improvement:

![](./Results/Protected/Shakespeare_LSTM_training_curve.png)

![](./Results/Protected/Shakespeare_RNN_training_curve.png)

Unfortunately, the character-level predictions made by the model are still completely incoherent even after 30 minutes of training. The following prediction is made by the RNN after 30 minutes (8600 batches) of training, given the prompt `"once upon a time"`:

```
once upon a timerrtirs!rfnhti todh eu yyn hp iee tmt tres  u.yd woeddiy aid ttdh ioi s ie ni? menle
 hicin
,r tnydvin is,ioe hrfmnfenei ?iyvlearb lelwamnndeken, n
 toeyeltymet himvotrlitarn dddeod mefts  oi.le wo ewusote
 toe lao-ms hhu yw
aliipfrienateo mpain ta
[
.rudo ssrek  oen mtsepeatlnooor; toywrd
 eow
 bl wlha  oneehhuw  aoioshhod.inlmf luyveelmnaw ay a ot to'dee ber hi gtllotsylad btw  se.scoe, al llresidl noal relythe  iia dsrey,' gouryt, bntia s., ;ueb r m.lfihad  ,gd, hoiiry,ynsb; sege,d.,u thlhmdo
```
