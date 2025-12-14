## What is a feature shift?
- A feature shift occurs when a given feature behaves differently in two different datasets. For example, we could have the feature blood pressure from two different hospital datasets, where hospital A could have calculated this data with a calibrated device while hospital's B device might be old. So it could happen that the mean is higher in Hospital B and the variance is larger. Here, a feature shift happens.

- Combining structured data from multiple sources can leat to mismatching and biased features, due to incorrect data collection procedures, human entry errors, faulty standardization, or erroneous data processing.

## Feature Shift Localization (FSL)
- Feature shift localization is the task of enumerating which features of multi-dimensional datasets are originating the distribution shift between two or more data sources
- Invariance to sample order (if you shuffle the samples inside a dataset X, the output should be the same)
- Equivariance to feature order (features are stored in a vector x1,...,xw. The output probability for each feature should not be affected by the order. For instance, in the MNIST dataset, x1 corresponds to the top left pixel because someone decided so. However, if we change the ordering and set x1 to the bottom right pixel the image still would be the same and the feature ordering should not affect probabilities. Since neural networks are sensitive to feature order, FSL-net give an output for each feature.)

### Related work
- AlthoughDataFix performs well in many cases, it struggles with detecting challenging feature shifts and scales poorly with high-dimensional and large datasets.
- Unlike model-centric approaches, which prioritize refining models while working with a fixed dataset, DCAI (Data-centric AI) emphasizes improving datasets through systematic and iterative processes.

## Problem definition
- We are given two datasets X and Y, each consisting of samples with the same d features.
- These datasets are drawn from two unknown distributions p (reference) and q (query).
- The goal is not just to detect that p and q are different, but to identify which features cause the difference.
- We define a set C containing the indices of the shifted or corrupted features. Removing these features from the data should eliminate the distribution shift. 3 conditions must hold:
    1. If I remove the corrupted features, the two datasets look statistically identical.
    2. The original distributions are different (ensure there is really a shift to detect)
    3. C must be minimal. **Find the smallest possible set of features whose removal explains the entire distribution shift.** 

# Network
- Overview: take as input reference and query datasets and output a probability vector representing in the i-th component the probability that feature i is corrupted. Then, C will contain the features having probability >0.5.
- Statistical Descriptor Network: vector summarizing input distributions.
- Prediction Network: takes both vectors and predicts corruption probabilities for each feature.

### A statistical functional map 
Summarizes a dataset as a matrix where each row corresponds to a feature and each column captures a statistical property of that feature or its interactions with others. (num_of_features x num_of_statistics)

## Statistical Descriptor Network
### 1. Statistical Measures
- Mean
- Standard Deviation
- Median
- Mean Absolute Deviation
- p-order moments
- Histogram
- Empirical CDF

All these measures work well for capturing marginal distributions but not for capturing correlations and higher order relations between dimensions.

### 2. Moment Extraction Network

Capture high order relations between features. 

Conv + BN + activation + mean-pooling --> d x t2 (t2 are output channels of the convolution)

### 3. Neural Embedding Network

Capture non-linear relationships between features. 

Concat X and Y --> conv + BN + residual block (conv + BN + activation + conv + BN + skip connections) --> divide into X and Y --> mean-pooling for each X and Y --> d x t3

## Prediction Network

Combines statistical feature maps from the statistical measures, moment extraction network, and neurla embedding network to predict the probabilities of each feature belonging to the corrupted set C.

Combination (normalized squared difference) of statistical feature maps into a single joint map (t x d) --> BN --> residual block --> conv --> sigmoid --> P_hat --> Binarization --> C_hat

**Loss function**: binary corss-entropy between predicted probabilities and the ground truth corrupted feature set + lambda*auxiliary function (applied to the predicted statistical functional maps)

**Properties**:

1. Sample-wise invariance
    - Shapes of maps are independent of sample size (thanks to mean pooling operations)
    - non-linear statistical functionals are computed sorting the dataset from smallest to largest, making them invariant to sample order
2. Feature-wise equivariance
    - Each of the statistical maps are computed using the marginal distributions
    - Each time a convolution is to be applied, the feature ordering is randomly shuffled so that the model is not sensitive to feature order
3. Locality (why do we want it? --> Sometimes neural networks when they see a change in a given feature they spread the signal everywhere, but we don't want that since we are analyzing feature shifts)
    - statistical measures are computed using marginals
    - addition of residual blocks in the nerual embedding network (you may add interactions with other features but you don't forget which feature you are)
    - adding auxiliary function (mixing unrelated features increases loss, the network is penalized for spreading the signal everywhere)


# FSL-Net vs DataFix
FSL-Net encodes each dataset into per-feature statistical descriptors (simple stats, moments, neural embeddings), then feeds the pairwise descriptor differences through a prediction network to output per-feature shift probabilities in a single forward pass. DataFix repeatedly trains a random forest to distinguish reference vs query, removes the most important features, and stops when a divergence criterion (with knee detection) suggests the shifted feature set has been identified. FSL-Net is better because it is trained once and then applied zero-shot, whereas DataFix must retrain multiple forests per dataset. It also scales much better to large sample sizes and high-dimensional data, while DataFix becomes slow and unstable in such regimes. Empirically, FSL-Net achieves higher localization accuracy and robustness across a wide range of shift types.
