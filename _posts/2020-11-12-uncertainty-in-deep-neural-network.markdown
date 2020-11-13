---
layout: post
title:  "Uncertainty in deep neural network"
date:   2020-11-12 20:00:00 +0700
categories: jekyll update
---
## Uncertainty in deep neural network

### 1. Introduction

#### Problem with current deep neural network

[[1]](#1) showed that modern (very) deep neural networks can achieve high predictive performances (acurracy), but they are poorly calibrated in terms of predicting corectness.

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/poor-calibration.JPG"|absolute_url}})

[[2]](#2) mathematically proved that neural networks equiped with ReLU activations yield unsensible softmax score for far-away data.

An intuitive example:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/cat-dog.png"|absolute_url}})

A deep NN classifier trained with dogs provides unexpected/unreliable output when being fed with cats. In other words, it doesn't know <em>where it knows</em>. This closed-world assumption makes deep neural network unaware of the cat category. Our expectation is that the model should output: "I guess it's Phu Quoc dog, but I am <em>not sure</em>".

We can measure the uncertainty of NN by <em>confidence score</em> (the `uncertainty` and `confidence` terms are antonyms and can be used interchangable).

Currently, the softmax score, i.e: the maximum element of the softmax vector of any deep NN-based classifier, is misinterpreted as the confidence score. It only give us a <em>relative probabilaties</em> between possible classes [[3]](#3), and in practice, it's unreliable measurement of NN's confidence level.

From above stories, we come to the conclusion that the modern NN doesn't master at provding a good/reliable confidence score for its predictions, that limits it applications in real world. In some high-risk applications, to push the usage of neural network to industrial level, meeting the user satisfaction, we should output a reliable level of uncertainty along with each AI-model's prediction. If the confidence score is high, we trust the AI-model, otherwise it output a low confidence score and the experts in this domain (human) will handle these cases.

#### Example of high-risk application:

Cancer detection on medical image: it's unacceptable if a cancer patient is mis-classified as a normal case. Even if the AI model has 99% recall of cancer class, there are still 1% of patient will die because of the AI.

Vehicle plate recognition: assume that we developed an AI vehicle plate recognition model for a car parking system with near-perfect accuracy (99%). But when applying it in real production, the 1% error rate can result in very terrible consequences, the customer can lost his car and we have to indemnify billions of VND!!!

In these above applications, the AI model is just a black-box system the receives the input image, produces an output and we don't have any clue to trust or not to trust it. So it is imperative for AI system to have a reliable confidence score. Take the vehicle plate recognition as an example, if the plate is blur, occluded, or it have strange text font/alphabet, i.e: the plate is Laotian plate, but the recognition model was trained on Vietnamese registration plates, the model should output a low confidence score, that means it is not certain about its prediction, and the system will emit a warning, and the human will double-check this plate to decide whether the AI model's output is correct or wrong. The idea can be visualized by the following figure:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/acc_conf_1.jpg"|absolute_url}})

In this figure, the horizontal axis represents confidence threshold, the vertical axis represents the accuracy. The red lines describe the relation between the confidence threshold and the <em>averaged accuracy of data sample whose confidence is larger than this corresponding confidence threshold</em>. Intuitively, we can choose a threshold to guarantee that the accuracy of the AI model is perfect (100%) or equal to a certain value (for e.g: 90%). If the model's prediction has lower confidence score than the threshold, the model will reject it and need the aid from human. In this example, if we choose confidence threshold of 0.5, the we can trust the model with 100% accuracy, and we only have to double-check the data sample with confidence score smaller the 0.5. Similarly, we can guarantee the accuracy of 90% at 0.1 confidence threshold.

On the contrary, the following figure illustrates an poorly calibrated confidence score behavior of an deep learning model.

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/acc_conf_2.jpg"|absolute_url}})

#### Defenition/Metric

Confidence score: a number in the range [0, 1], representing the level of confidence the neural network has for each of its predictions.

Metric: we can use ECE/MCE or AUC.

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