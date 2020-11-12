---
layout: post
title:  "Welcome to Jekyll!"
date:   2020-11-12 20:00:00 +0700
categories: jekyll update
---
## Uncertainty in deep neural network

### 1. Introduction

#### Problem with current deep neural network

[[1]](#1) showed that modern (very) deep neural networks can achieve high predictive performances (acurracy), but they are poorly calibrated in terms of predicting corectness.

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/poor-calibration.jpg"|absolute_url}})

[[2]](#2) mathematically proved that neural networks equiped with ReLU activations yield unsensible softmax score for far-away data.

An intuitive example:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/cat-dog.png"|absolute_url}})

A trained classifier train with dogs provide unexpected/unreliable output when fed with cats. In other words, it doesn't know <em>where it know</em>. Our expectation is that the model should output: "I guess it's Phu Quoc dog, but I am <em>not sure</em>".

We can measure the uncertainty of NN by <em>confidence score</em> (the `uncertainty` and `confidence` terms are antonyms and can be used interchangable).

Currently, the softmax score, i.e: the maximum element of the softmax vector of any deep NN-based classifier, is misinterpreted as the confidence score. It only give us a <em>relative probabilaties</em> between possible classes [[3]](#3), and in practice, it's unreliable measurement of NN's confidence level.

From above stories, we come to the conclusion that the modern NN doesn't master at provding a good/reliable confidence score for its predictions, that limits it applications in real world. In some high-risk applications, to push the usage of neural network to industrial level, meeting the user satisfaction, we should output a reliable level of uncertainty along with each AI-model's prediction. If the confidence score is high, we trust the AI-model, otherwise it output a low confidence score and the experts in this domain (human) will handle these cases.

#### Example of high-risk application:

Cancer detection on medical image

Vehicle plate recognition

#### Defenition/Metric

It is a number in the range [0, 1], representing the level of confidence the neural network has for each of its predictions.

For example:

Metric: we can use ECE/MCE or AUC

### 2. Types of uncertainties

- Epistemic uncertainty

- Aleatoric uncertainty

### 3. Approaches

- Bayesian/Variational inference method

- Out-of-distribution detection

- Confidence score predictor

- Distance-based confidence score

### 4. Conclusions

### References
<a id="1">[1]</a> 
Guo, Chuan, et al. "On calibration of modern neural networks." arXiv preprint arXiv:1706.04599 (2017). 

<a id="2">[2]</a> 
Hein, Matthias, Maksym Andriushchenko, and Julian Bitterwolf. "Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

<a id="3">[3]</a> 
Kendall, Alex, Vijay Badrinarayanan, and Roberto Cipolla. "Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding." arXiv preprint arXiv:1511.02680 (2015).