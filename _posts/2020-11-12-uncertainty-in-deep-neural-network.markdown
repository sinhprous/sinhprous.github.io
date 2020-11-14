---
layout: post
title:  "Uncertainty in deep neural network"
date:   2020-11-12 20:00:00 +0700
categories: jekyll update
---
## Uncertainty in deep neural network

Please note that this post is for my own educational purpose.

### 1. Introduction

#### Problem with current deep neural network

[[1]](#1) showed that modern (very) deep neural networks can achieve high predictive performances (acurracy), but they are poorly calibrated in terms of predicting corectness.

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/poor-calibration.JPG"|absolute_url}})

[[2]](#2) mathematically proved that neural networks equipped with ReLU activations yield unsensible softmax score for far-away data.

An intuitive example:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/cat-dog.png"|absolute_url}})

A deep NN classifier trained with dogs provides unexpected/unreliable output when being fed with cats. In other words, it doesn't know <em>where it knows</em>. This closed-world assumption makes deep neural network unaware of the cat category. Our expectation is that the model should output: "I guess it's Phu Quoc dog, but I am <em>not sure</em>".

We can measure the uncertainty of NN by <em>confidence score</em> (the `uncertainty` and `confidence` terms are antonyms and can be used interchangable).

Currently, the softmax score, i.e: the maximum element of the softmax vector of any deep NN-based classifier, is misinterpreted as the confidence score. It only give us a <em>relative probabilaties</em> between possible classes [[3]](#3), and in most practical cases, it's unreliable measurement of NN's confidence level (*).

---
**NOTE**

(*) Sometimes, the performance of advanced techniques for estimating confidence score is not as good as the original softmax score. [[4]](#4) showed that for object detection problem, the performance of original softmax score surpassed the MC-Dropout method that we will cover later.

---

From above stories, we come to the conclusion that the modern NN doesn't master at provding a good/reliable confidence score for its predictions, that limits it applications in real world. In some high-risk applications, to push the usage of neural network to industrial level, meeting the user satisfaction, we should output a reliable level of uncertainty along with each AI-model's prediction. If the confidence score is high, we trust the AI-model, otherwise it output a low confidence score and the experts in this domain (human) will handle these cases.

#### Example of high-risk application:

Cancer detection on medical image: it's unacceptable if a cancer patient is mis-classified as a normal case. Even if the AI model has 99% recall of cancer class, there are still 1% of patient will die because of the AI.

Vehicle plate recognition: assume that we developed an AI vehicle plate recognition model for a car parking system with near-perfect accuracy (99%). But when applying it in real production, the 1% error rate can result in very terrible consequences, the customer can lost his car and we have to indemnify billions of VND!!!

In these above applications, the AI model is just a black-box system the receives the input image, produces an output and we don't have any clue to trust or not to trust it. So it is imperative for AI system to have a reliable confidence score. Take the vehicle plate recognition as an example, if the plate is blur, occluded, or it have strange text font/alphabet, i.e: the plate is Laotian plate, but the recognition model was trained on Vietnamese registration plates, the model should output a low confidence score, that means it is not certain about its prediction, and the system will emit a warning, and the human will double-check this plate to decide whether the AI model's output is correct or wrong. The idea can be visualized by the following figure:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/acc_conf_1.jpg"|absolute_url}})

In this figure, the horizontal axis represents confidence threshold, the vertical axis represents the accuracy. The red lines describe the relation between the confidence threshold and the <em>averaged accuracy of data sample whose confidence is larger than this corresponding confidence threshold</em>. Intuitively, we can choose a threshold to guarantee that the accuracy of the AI model is perfect (100%) or equal to a certain value (for e.g: 90%). If the model's prediction has lower confidence score than the threshold, the model will reject it and need the aid from human. In this example, if we choose confidence threshold of 0.5, then we can trust the model with 100% accuracy, and we only have to double-check the data sample with confidence score smaller the 0.5. Similarly, we can guarantee the accuracy of 90% at 0.1 confidence threshold.

On the contrary, the following figure illustrates a poorly calibrated confidence score behavior of a deep learning model.

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/acc_conf_2.jpg"|absolute_url}})

#### Defenition/Metric

Confidence score: a number in the range [0, 1], representing the level of confidence the neural network has for each of its predictions.

Metric: we can use ECE/MCE or AUC.

- ECE: weighted sum of the red areas in the figure 1. ECE represent the difference between the accuracy and confidence level. We would like the confidence to represent the true probability of correctness. Intuitively, if the model infers on 100 samples, each with confidence score of 0.8, then the expected accuracy for these samples should be 80%.

- AUC: we also would like the confidence score to be an indicator to classify the correct prediction and wrong prediction of a DNN. Basically, if the confidence score is high, the corresponding prediction should be correct, and vice versa, if the confidence score is low, the corresponding prediction is more likely to be wrong. So we can think of confidence score as the score of a binary classification problem, and we can use the AUC metric to evaluate the performance of the confidence score of a DNN model.

### 2. Types of uncertainties

- Epistemic uncertainty: the uncertainty caused by the uncertainty of the model's weights (insufficient knowledge about which parameters best model the data). We can reduce this type of uncertainty by giving more data to the model.

- Aleatoric uncertainty: the uncertainty caused by the characteristic of the data (for e.g: sensor noise) and can not be reduced even if we feed more data to the neural network.

### 3. Approaches

#### Bayesian/Variational inference method

- Monte carlo Dropout (MC Dropout)

Normally, we used maximum likelihood to optimize the cost function of a neural network and finally end up with the fixed/deterministic model's weights (parameters). Instead of providing point estimatation of model's parameters, Bayesian NN method tries to provide a posterior distribution over model's parameters, given the training data: <img src="https://latex.codecogs.com/gif.latex?p(\omega|X, Y)\propto p(Y|X,\omega)p(\omega)" />.

Given posterior distribution over model's parameters, we can obtain the predictive distribution as follows:

<img src="https://latex.codecogs.com/gif.latex?p(y|x, X, Y)=\int p(y|x, \omega)p(\omega|X, Y)d\omega" />

The true posterior distribution over model's weight is analytically intractable, so MC Dropout [[5]](#5) approximates it by a variational distribution <img src="https://latex.codecogs.com/gif.latex?q(\omega)" />. This distribution is defined as:

<img src="https://latex.codecogs.com/gif.latex?W_i=M_i \cdot \mathrm{diag}([z_{i,j}]_{j=1}^{K_i}))" />
<br/><br/>
<img src="https://latex.codecogs.com/gif.latex?z_{i,j} \sim \mathrm{Bernoulli}(p_i) \text{ for } i=1, ..., L, j=1, ..., K_{i-1}" />

Where <img src="https://latex.codecogs.com/gif.latex?M_i" /> is variational parameter, <img src="https://latex.codecogs.com/gif.latex?i" /> indicates the index of layer, <img src="https://latex.codecogs.com/gif.latex?j" /> represents the index of neural unit in each layer. From these above formulas, we realize that this is exactly the form of dropout applied in the <img src="https://latex.codecogs.com/gif.latex?i" />-th layer's input: The columns of <img src="https://latex.codecogs.com/gif.latex?M_i" /> is randomly replaced by zero vector, with the probability <img src="https://latex.codecogs.com/gif.latex?p_i" />.

Now, the Bayesian approximation's objective is to minimize the KL divergence between the approximated distribution <img src="https://latex.codecogs.com/gif.latex?q(\omega)" /> and the true distribution <img src="https://latex.codecogs.com/gif.latex?p(\omega|X, Y)" />. This KL divergence can be write as follows:

<img src="https://latex.codecogs.com/gif.latex?-\int q(\omega) \log p(Y|X, \omega)d\omega + \mathrm{KL} (q(\omega)||p(\omega))" />.

The paper [[5]](#5) and its appendix [[6]](#6) mathematically proved that this objective function is equal to the normal loss function of standard neural network equipped with dropout:

<img src="https://latex.codecogs.com/gif.latex?L_{\text{dropout}} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{E}(y_i, \hat{y_i})+\lambda\sum_{i=1}^{L}(||W_i||^2_2+||b||^2_2)" />.

So, we come to the conclusion that training a NN with dropout is mathematically equal to approximated inference of Bayesian model.

Now, we perform moment-matching and estimate the first two moments of the predictive distribution above empirically:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/1st_moment.jpg"|absolute_url}})

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/2nd_moment.jpg"|absolute_url}})

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/var.jpg"|absolute_url}})

This Monte Carlo estimatation is referred as MC dropout by the author. 

From pratical point of view, in test time, we just <b>keep the dropout enabled</b>, and perform T stochastic forward passes through the network. To obtain the final prediction we just average the results, and to obtain the model uncertainty we just <em>calculate sample variance of these results</em>. A good tutorial for MC dropout can be found at here https://github.com/xuwd11/Dropout_Tutorial_in_PyTorch.

Example result for regression problem:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/mcdo_out.jpg"|absolute_url}})

- MC Batchnorm

[[7]](#7) used the similar variational approximation as MC Dropout, but the source of randomness comes from batch normalization (BN) layer instead of dropout operation. 

When training NN with BN, the inference at training time for a sample <img src="https://latex.codecogs.com/gif.latex?x" /> is a stochastic process, varying based on other samples in the mini-batch (which we use to calculate units' means and deviations). When performing inference at test time with standard Batchnorm, the BN units' means and standard deviations are calculated from <em>training dataset</em>, but in MC Batchnorm, we calculate these means and deviations <em>from the minibatches</em>.

The detailed algorithm:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/mc_bn_alg.jpg"|absolute_url}})

- Depth uncertainty:

Different from MC Dropout and MC Batchnorm, which requires a large number <img src="https://latex.codecogs.com/gif.latex?T" /> stochastic forward passes to get stable results, this method [[8]](#8) compute exact predictive posteriors with a <em>single forward pass</em>. Depth uncertainty treats the depth of a Neural Network as a random variable over which to perform inference. The authors placed a categorical distribution over the depth of a neural network. The architecture of Depth uncertainty is shown here:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/depth_unc.jpg"|absolute_url}})

Basically, The output block is applied to each intermediate layer's activations.

Predictions for new data <img src="https://latex.codecogs.com/gif.latex?x" /> are made by marginalising depth with the variational posterior:

![figure]({{"/asset/2020-11-12-uncertainty-in-deep-neural-network/depth_unc_inf.jpg"|absolute_url}})

This class of methods estimates the posterior distribution over the model's weights/depth, so it only produces `epistemic` uncertainty.

#### Out-of-distribution detection

TBD

#### Confidence score predictor

TBD

#### Distance-based confidence score

TBD

### 4. Conclusions

Reliable confidence score estimation is a good toolkit to monitor and judge the output of any deep neural network model. It paves the way for AI applications to be more trustable, interpretable, explainable and become more widely applied and deployed in real world.

### References
<a id="1">[1]</a> 
Guo, Chuan, et al. "On calibration of modern neural networks." arXiv preprint arXiv:1706.04599 (2017). 

<a id="2">[2]</a> 
Hein, Matthias, Maksym Andriushchenko, and Julian Bitterwolf. "Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

<a id="3">[3]</a> 
Kendall, Alex, Vijay Badrinarayanan, and Roberto Cipolla. "Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding." arXiv preprint arXiv:1511.02680 (2015).

<a id="4">[4]</a>
Beckersjürgen, Yannik. Uncertainty Estimation for Object Detection. Diss. 2019.

<a id="5">[5]</a>
Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. 2016.

<a id="6">[6]</a>
Gal, Y., and Z. Ghahramani. "Dropout as a Bayesian Approximation: Appendix 20 (2016)." URL http://arxiv. org/abs/1506.02157 1506.

<a id="7">[7]</a>
Teye, Mattias, Hossein Azizpour, and Kevin Smith. "Bayesian uncertainty estimation for batch normalized deep networks." arXiv preprint arXiv:1802.06455 (2018).

<a id="8">[8]</a>
Antorán, Javier, James Allingham, and José Miguel Hernández-Lobato. "Depth uncertainty in neural networks." Advances in Neural Information Processing Systems 33 (2020).