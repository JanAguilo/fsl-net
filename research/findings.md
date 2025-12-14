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

### 2. Moment Extraction Network

### 3. Neural Embedding Network


