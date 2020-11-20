# CSCI596-FedLearningProject
## Goal: Try to solve data imbalance problem in Federated Learning

### Background:
**Federated Learning**:

Federated learning (also known as collaborative learning) is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. This approach stands in contrast to traditional centralized machine learning techniques where all the local datasets are uploaded to one server, as well as to more classical decentralized approaches which often assume that local data samples are identically distributed.
![Federated Learning](https://github.com/ivishalanand/Federated-Learning-on-Hospital-Data/raw/master/images/federated-learning.png)


**Data imbalance**:

1) Size Imbalance, where the data size on each device (or client) is uneven; 
2) Local Imbalance, i.e., independent and non-identically distribution (non-IID), where each device does not follow a common data distribution; 
3) Global Imbalance, means that the collection of data in all devices is class imbalanced

**Proposed solution**:

Paper Link: https://ieeexplore.ieee.org/document/9141436

Model Framework:

<img width="600" src="https://user-images.githubusercontent.com/17812876/99761073-0f093280-2aaa-11eb-8b0c-6dbac69a9383.png" alt="modelFramework">

Algorithm Framework:

<img width="500" src="https://user-images.githubusercontent.com/17812876/99761162-41b32b00-2aaa-11eb-94db-a7376aa071bd.png" alt="algFramework">

Workflow:
1)Initialization: The FL server initializes the weights and the optimizer of the neural network model and collects the local data distribution of participants.
2)Rebalancing: First, we perform the z-score-based data augmentation and downsampling to relieve the global imbalanced of training data. Then, we propose the mediator which asynchronouly receives and applies the updates from clients to averaging the local imbalance. 
3)Training: First, each mediator sends the model to the subordinate clients. Then each client trains the model for E local epochs and returns the updated model to the corresponding mediator. Loops this process Em times. Finally, all the mediators send the updates of models to the FL server.
4)Aggregation: The FL server aggregates all the updates using the FedAvg algorithm.

**Benchmark and Dataset**:

https://github.com/chaoyanghe/Awesome-Federated-Learning#Benchmark-and-Dataset

**Baseline Code (just examples)**:
- https://github.com/shaoxiongji/federated-learning

- https://github.com/ivishalanand/Federated-Learning-on-Hospital-Data


### Steps:
- Replicate Baseline Code
- Run evaluation pipeline
- Improve the baseline by https://docs.google.com/document/d/1qa0Cv-axRw9ZSVDB-C5YZMp9lEd6qp3d1FmzX7D8Mgg/edit methods
- Propose our own method (optional)
