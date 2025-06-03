# Catastrophic Forgetting Research

A comprehensive research framework for exploring catastrophic forgetting in neural networks, featuring both Python and JavaScript implementations with interactive visualizations.

## Overview

Catastrophic forgetting occurs when a neural network completely loses previously learned knowledge upon learning new tasks. This repository provides tools and visualizations to study this phenomenon in detail, making it accessible for both research and educational purposes.

## Interactive Visualization

### Quick Start

1. **Open the visualization:**
   ```bash
   # Navigate to the JavaScript source directory
   cd javascript/src/
   
   # Open the HTML file in your web browser
   open catastrophic_forgetting_viz.html
   # or on Linux: xdg-open catastrophic_forgetting_viz.html
   # or on Windows: start catastrophic_forgetting_viz.html
   ```

2. **Alternative: Direct browser access**
   - Simply open `javascript/src/catastrophic_forgetting_viz.html` in any modern web browser
   - No server setup required - runs entirely in the browser

### How to Use the Visualization

The demonstration follows a 3-phase process that clearly shows catastrophic forgetting:

#### Phase 1: Initial Learning
1. Click **"Train on Dataset 1"** button
2. Watch the classifier learn to separate red and blue points
3. Observe the decision boundary (black line) form between the two classes
4. Note the accuracy reaching ~100% after 50 training epochs

#### Phase 2: New Task Learning
1. Click **"Train on Dataset 2"** button (enabled after Phase 1)
2. The classifier now trains on a different dataset with points in new locations
3. Watch the decision boundary move to accommodate the new data
4. Accuracy again reaches ~100% for the new task

#### Phase 3: Testing Original Knowledge
1. Click **"Test Original Data"** button (enabled after Phase 2)
2. The visualization now shows only the original Dataset 1 points
3. **Observe catastrophic forgetting**: Points that were correctly classified before now show ✗ marks
4. The accuracy drops significantly (often to 0-25%), demonstrating complete knowledge loss

### Understanding the Interface

**Visualization Area:**
- **Canvas**: Shows data points, decision boundary, and background coloring
- **Red points**: Class 0 samples
- **Blue points**: Class 1 samples
- **Black line**: Current decision boundary
- **Background**: Color intensity shows prediction confidence

**Controls Panel:**
- **Phase Indicator**: Shows current training phase with color coding
- **Training Controls**: Step through the 3-phase demonstration
- **Reset**: Return to initial state to try again

**Metrics Display:**
- **Epoch**: Current training iteration
- **Accuracy**: Classification performance (color-coded: green=high, orange=medium, red=low)
- **Loss**: Training loss value
- **Weights & Bias**: Current model parameters

### Key Observations

- **Perfect Initial Learning**: The classifier achieves 100% accuracy on Dataset 1
- **Successful New Learning**: The classifier achieves 100% accuracy on Dataset 2
- **Complete Forgetting**: When tested on Dataset 1, accuracy drops to 0-25%
- **Decision Boundary Shift**: The boundary completely moves to fit new data, abandoning old knowledge

This visualization demonstrates why catastrophic forgetting is a critical problem in continual learning scenarios.