# Garbage Classification using CNN

A deep learning project for classifying garbage images into 6 categories: cardboard, glass, metal, paper, plastic, and trash. The model is built using PyTorch Lightning with a custom CNN architecture optimized through Bayesian hyperparameter search.

## Dataset

- **Classes**: 6 (cardboard, glass, metal, paper, plastic, trash)
- **Image Size**: 128×128 pixels
- **Split**: train/validation/test

## Model Architecture

Custom SimpleCNN with the following configurable components:
- Convolutional layers with flexible feature strategies (same, doubling, halving)
- Batch normalization support
- Multiple activation functions (ReLU, GELU, SiLU, Mish)
- Dropout regularization
- Max pooling
- Dense layers with configurable neurons

**Best Model Configuration:**
- **Total Parameters**: 547,334 (all trainable)
- **Model Size**: 2.19 MB
- **Base Features**: 64
- **Layers**: 5 convolutional layers
- **Filter Size**: 5×5
- **Strategy**: same (constant feature maps across layers)
- **Conv Activation**: Mish
- **Dense Neurons**: 128
- **Dense Activation**: ReLU
- **Batch Normalization**: Enabled
- **Dropout**: 0.2

## Hyperparameter Optimization

### Search Strategy

Bayesian optimization with early termination (Hyperband) to efficiently explore the hyperparameter space.

**Swept Hyperparameters:**
- `base_features`: [16, 32, 64]
- `strategy`: ['same', 'doubling', 'halving']
- `filter_size`: [3, 5]
- `conv_activation`: ['relu', 'gelu', 'silu', 'mish']
- `dense_activation`: ['relu', 'gelu', 'silu', 'mish']
- `num_dense`: [64, 128, 256, 512]
- `include_batchnorm`: [true, false]
- `dropout`: [0.2, 0.3, 0.4, 0.5]
- `lr`: log-uniform [0.0001, 0.01]
- `batch_size`: [16, 32, 64]
- `use_augmentation`: [true, false]

**Fixed Parameters:**
- Max epochs: 20
- Early stopping: Hyperband (min_iter=5)
- Augmentation: Finetuning*

### Optimization Strategy

1. **Bayesian Optimization**: Instead of random or grid search, used Bayesian optimization to intelligently sample the hyperparameter space based on previous results
2. **Early Termination**: Implemented Hyperband to stop poorly performing runs after 5 epochs, reducing computational cost
3. **Run Cap**: Limited to 50 runs to balance exploration and resource usage
4. **Metric**: Maximized validation accuracy

### Correlation Analysis

| Hyperparameter | Correlation with val_accuracy |
|---------------|------------------------------|
| lr            | -0.330                       |
| dropout       | -0.239                       |
| num_dense     | -0.100                       |
| base_features | +0.011                       |

**Key Insights:**
- **Learning Rate**: Strong negative correlation (-0.33) suggests that lower learning rates (around 0.0001) perform better for this task
- **Dropout**: Moderate negative correlation (-0.24) indicates that lower dropout rates are preferred, with 0.2 being optimal
- **Dense Neurons**: Weak negative correlation suggests diminishing returns beyond 128 neurons
- **Base Features**: Near-zero correlation indicates that model capacity at the convolutional level is less critical than other factors

## Results

### Best Run: `expert-sweep-2`

**Training Performance (Epoch 19):**
- Train Accuracy: **95.98%**
- Train Loss: 0.134

**Per-Class Training F1-Scores:**
- Cardboard: 0.973
- Paper: 0.980
- Plastic: 0.953
- Glass: 0.951
- Trash: 0.944
- Metal: 0.942

**Validation Performance:**
- Validation Accuracy: **80.04%**
- Validation Loss: 0.667

**Per-Class Validation F1-Scores:**
- Paper: 0.875
- Cardboard: 0.872
- Metal: 0.808
- Trash: 0.769
- Glass: 0.751
- Plastic: 0.709

### Test Set Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **85.46%** |
| Test Loss | 0.409 |

**Per-Class Test Metrics:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cardboard | 0.914 | 0.925 | 0.919 |
| Paper | 0.972 | 0.883 | 0.926 |
| Glass | 0.779 | 0.907 | 0.838 |
| Plastic | 0.790 | 0.865 | 0.826 |
| Metal | 0.853 | 0.725 | 0.784 |
| Trash | 0.769 | 0.690 | 0.727 |

## Key Observations

### 1. Generalization Gap
The model shows a ~10% performance gap between training (95.98%) and validation (80.04%), indicating some overfitting. However, test accuracy (85.46%) exceeds validation accuracy, suggesting the model generalizes reasonably well.

### 2. Class-Specific Performance
- **Best Performance**: Paper (F1: 0.926) and Cardboard (F1: 0.919) are classified most accurately
- **Challenging Classes**: Trash (F1: 0.727) and Metal (F1: 0.784) are harder to classify, likely due to visual similarity with other categories

### 3. Activation Functions
Mish activation for convolutional layers paired with ReLU for dense layers provided optimal performance, suggesting that smoother activations benefit feature extraction while standard ReLU suffices for classification.

### 4. Architecture Design
The 'same' strategy (constant feature maps) with 64 base features proved sufficient, indicating that aggressive feature expansion is not necessary for this dataset size.

### 5. Regularization Trade-offs
Lower dropout (0.2) performed best despite the generalization gap, suggesting that the model benefits from retaining more information during training rather than aggressive regularization.

## Finetuning with Pretrained Models

We further improved performance by finetuning a pre-trained **MobileNetV3 Small** model.

### Best Run: `wild-sweep-6`
- **Validation Accuracy**: **98.17%**
- **Hyperparameters**:
  - `lr`: 0.000398
  - `batch_size`: 64
  - `img_size`: 224
  - `freeze_strategy`: `finetune_all`
  - `backbone_lr_factor`: 0.1

### Test Set Performance (Finetuned)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **98.41%** |
| Test Loss | 0.075 |

**Per-Class Test Metrics:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cardboard | 1.000 | 1.000 | 1.000 |
| Paper | 0.984 | 1.000 | 0.992 |
| Glass | 0.980 | 1.000 | 0.990 |
| Metal | 0.987 | 0.975 | 0.981 |
| Plastic | 0.969 | 0.990 | 0.979 |
| Trash | 1.000 | 0.828 | 0.906 |

## Model Comparison

| Model | Test Accuracy | Best F1 Class | Worst F1 Class |
|-------|---------------|---------------|----------------|
| SimpleCNN (Scratch) | 85.46% | Paper (0.93) | Trash (0.73) |
| **MobileNetV3 (Finetuned)** | **98.41%** | **Cardboard (1.00)** | **Trash (0.91)** |

## Sample Predictions

![Test Predictions Grid](notebooks/test_predictions_grid.png)

*10×3 grid showing sample test images with ground truth and predictions. Green titles indicate correct predictions, red indicates misclassifications.*

## Future Work

- Experiment with Larger Models (ResNet50, EfficientNetV2)
- Advanced data augmentation strategies
- Ensemble methods
- Addressing class imbalance if present

## Requirements

```
pytorch-lightning
torch
torchvision
wandb
scikit-learn
matplotlib
pandas
torchinfo
```

## Usage

### Training
```bash
wandb sweep config.yaml
wandb agent 23f2001173-indian-institute-of-technology-madras/garbage_clf-src_scripts/rzpn3bhn --count 50
```

### Inference
```python
from src.lightning.model import ModelLightning
from src.models.simple_cnn import SimpleCNN

model = SimpleCNN(...)
pl_model = ModelLightning.load_from_checkpoint(
    'checkpoints/best_model_epoch=12_val_accuracy=0.8049.ckpt',
    model=model,
    idx_to_class=idx_to_class
)
```

## Project Structure

```
garbage_clf/
├── config.yaml              # Hyperparameter sweep configuration
├── checkpoints/             # Model checkpoints
├── data/                    # Dataset (train/val/test)
├── notebooks/              # Analysis and visualization notebooks
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── lightning/          # PyTorch Lightning modules
│   ├── models/             # Model architectures
│   └── scripts/            # Training scripts
└── wandb/                  # Weights & Biases logs
```

## License

MIT

---

*Developed using PyTorch Lightning and optimized with Weights & Biases*
