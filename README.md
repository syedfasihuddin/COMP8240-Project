# Multi-Label text classification
Group project for COMP8240 Applications of data science

## Getting Started

With continuous increase in available data, there is a pressing need to organize it and modern classification problems often involve the prediction of multiple labels simultaneously associated with a single instance.
Known as Multi-Label Classification, it is one such task which is omnipresent in many real world problems.

In this project, using a Kaggle problem as example, we explore different aspects of multi-label classification.

### Bird’s-eye view of the project:
* Part-1: Overview of Multi-label classification.
* Part-2: Problem definition & evaluation metrics.
* Part-3: Exploratory data analysis (EDA).
* Part-4: Data pre-processing.
* Part-5: Multi-label classification techniques.

For a more detailed description you may refer to this [Page](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff)

### Techniques used for Solving a Multi-Label classification problem
Basically, there are three methods to solve a multi-label classification problem, namely:

* Problem Transformation
* Adapted Algorithm

#### Problem Transformation
In this method, we will try to transform our multi-label problem into single-label problem.

This method can be carried out in three different ways as:

* Binary Relevance
* Classifier Chains
* Label Powerset

#### Adapted Algorithm
Here we use a multi-label lazy learning approach named ML-KNN which is derived from the
traditional K-nearest neighbor (KNN) algorithm.The skmultilearn.adapt module implements
algorithm adaptation approaches to multi-label classification, including but not limited to
ML-KNN.


## Contributors

* **Fasihuddin** 
* [**Talha Islam**](https://github.com/TalhaIslam)
* [**Muhammad Hassaan**](https://github.com/hassan25sohail)
