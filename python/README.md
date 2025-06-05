# Python MLP Catastrophic Forgetting Demonstration

This directory contains a Python implementation of a Multi-Layer Perceptron (MLP) that demonstrates catastrophic forgetting in neural networks.

## Files

- `mlp_4class_forgetting.py` - Main interactive script demonstrating catastrophic forgetting
- `mlp_4class_forgetting_demo.py` - Non-interactive version for automated runs
- `test_mlp.py` - Test suite for the MLP implementation
- `requirements.txt` - Python dependencies (none - pure Python implementation)

## What is Catastrophic Forgetting?

Catastrophic forgetting occurs when a neural network forgets previously learned tasks when learning new tasks. This implementation demonstrates this phenomenon by:

1. **Task 1**: Training the network to classify Red (Class 0) and Green (Class 1) points
2. **Task 2**: Training the network to classify Blue (Class 2) and Yellow (Class 3) points

After Task 2 training, the network completely forgets how to classify Task 1 classes.

## Features

- **Pure Python Implementation**: No external dependencies like PyTorch or TensorFlow
- **4-Class Classification**: Demonstrates forgetting across multiple classes
- **Interactive Training**: Pauses between training phases for user input
- **Detailed Statistics**: Shows loss, accuracy, and weight magnitudes during training
- **Comprehensive Testing**: Includes unit tests for all components

## Usage

### Interactive Version
```bash
python mlp_4class_forgetting.py
```

This version will:
1. Train on Task 1 (Classes 0 & 1) and show progress
2. Pause and ask if you want to continue
3. Train on Task 2 (Classes 2 & 3) and show progress  
4. Pause and ask if you want to test forgetting
5. Test how well the network remembers Task 1
6. Display a summary of the catastrophic forgetting

### Non-Interactive Demo
```bash
python mlp_4class_forgetting_demo.py
```

This version runs all phases automatically without pausing for user input.

### Run Tests
```bash
python test_mlp.py
```

## Network Architecture

- **Input Layer**: 2 neurons (x, y coordinates)
- **Hidden Layer**: 8 neurons with sigmoid activation
- **Output Layer**: 4 neurons with softmax activation (one per class)
- **Training**: Backpropagation with cross-entropy loss

## Dataset

The 4-class dataset consists of points in different quadrants:

- **Class 0 (Red)**: Top-left quadrant (around x=1-2.5, y=6-7.5)
- **Class 1 (Green)**: Bottom-right quadrant (around x=6-7.5, y=1-2.5)  
- **Class 2 (Blue)**: Top-right quadrant (around x=6-7.5, y=6-7.5)
- **Class 3 (Yellow)**: Bottom-left quadrant (around x=1-2.5, y=1-2.5)

Each task contains 16 training samples (8 per class).

## Expected Results

A typical run shows severe catastrophic forgetting:

```
Task 1 accuracy BEFORE Task 2: 100.0%
Task 1 accuracy AFTER Task 2:  0.0%
Task 2 accuracy:                100.0%

Catastrophic forgetting: 100.0% drop in Task 1 performance!
🔴 Severe catastrophic forgetting observed!
```

## Implementation Details

### Key Classes and Functions

- `MLP4ClassClassifier`: Main neural network class
- `generate_task1_dataset()`: Creates Task 1 training data (Classes 0 & 1)
- `generate_task2_dataset()`: Creates Task 2 training data (Classes 2 & 3)
- `test_task_knowledge()`: Evaluates network performance on specific tasks

### Training Process

1. **Forward Pass**: Input → Hidden (sigmoid) → Output (softmax)
2. **Loss Calculation**: Cross-entropy loss between predictions and true labels
3. **Backward Pass**: Compute gradients using backpropagation
4. **Weight Update**: Apply gradients with learning rate

### Statistics Tracked

- **Loss**: Cross-entropy loss per epoch
- **Accuracy**: Classification accuracy per epoch
- **Weight Magnitudes**: L2 norm of weight matrices
- **Per-Class Performance**: Individual class accuracies

## Relationship to JavaScript Version

This Python implementation mirrors the JavaScript version in `../javascript/src/mlp_4class_classifier.js` but with these differences:

- **Command-line interface** instead of web-based visualization
- **Interactive pausing** between training phases
- **Detailed console output** with statistics
- **Pure Python** implementation without external dependencies

Both versions demonstrate the same catastrophic forgetting phenomenon using identical network architectures and datasets.