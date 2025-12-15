# Scientific Report and Model Explainability

This report details the scientific findings and explainability analysis of our Garbage Classification models. It includes visualizations of feature maps, guided backpropagation results, and hyperparameter optimization studies.

## Table of Contents
- [1. Feature Map Analysis](#1-feature-map-analysis)
- [2. Model Explainability: Guided Backpropagation](#2-model-explainability-guided-backpropagation)
- [3. Hyperparameter Analysis and Performance](#3-hyperparameter-analysis-and-performance)
  - [3.1. Model Trained from Scratch](#31-model-trained-from-scratch)
  - [3.2. Fine-Tuned Model](#32-fine-tuned-model)

## 1. Feature Map Analysis

In Convolutional Neural Networks (CNNs), the initial layers typically function as low-level feature extractors. Visualizing these layers helps confirm that the model is learning meaningful patterns from the data.

![First Layer Feature Maps](notebooks/first%20conv%20layer%20feat%20maps.png)

**Findings:**
*   **Edge and Boundary Detection:** The feature maps from the first convolutional layer demonstrate a strong focus on detecting edges, curves, and boundaries.
*   **Structural Foundations:** This observation confirms that the early layers are successfully establishing the structural basis required for more complex feature extraction in subsequent layers.

## 2. Model Explainability: Guided Backpropagation

To interpret the internal representations of the network, I employed Guided Backpropagation. This method visualizes which parts of the input image activate specific neurons in deeper layers.

![Guided Backprop (CONV5)](notebooks/guided%20backprop%20(CONV5).png)

**Findings:**
*   **Neuron Specialization:** The visualizations for the fifth convolutional layer (CONV5) show that different neurons focus on distinct parts of the image.
*   **Semantic Separation:** The model effectively distinguishes valid object features from background noise, indicating a robust hierarchical representation of the garbage classes.

## 3. Hyperparameter Analysis and Performance

I utilized Weights & Biases to track and visualize the impact of various hyperparameters on model performance. The following plots illustrate the results for both the model trained from scratch and the fine-tuned model.

### 3.1. Model Trained from Scratch

![Parallel Coordinates - Scratch](static/parallel_cord_scratch.png)
![Validation Accuracy Distribution - Scratch](static/val_acc-vs-createde-scratch.png)

**Observations:**
A batch size of 32 appears to be more effective, particularly when using Mish activation in the convolutional blocks, while the dense layers tend to favor ReLU activation. Dense layer configurations across different ranges performed comparatively better when using the `same` strategy. A dropout rate of 0.2 was selected in most high-performing configurations. Overall, higher validation performance was achieved through combinations of hyperparameters, rather than any single dominant parameter.

### 3.2. Fine-Tuned Model

![Parallel Coordinates - Fine-Tuned](static/parallel_cord_ft.png)
![Validation Accuracy Distribution - Fine-Tuned](static/val_acc-vs-created-ft.png)

**Observations:**
The fine-tuning experiments demonstrate a more consistent performance profile. Freezing half the layers yeilded the best results. Using differential lr is preferred. The plots indicate that leveraging pre-trained weights significantly stabilizes the training process, leading to higher average validation accuracy compared to the scratch model.
