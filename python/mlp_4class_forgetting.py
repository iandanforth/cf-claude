#!/usr/bin/env python3
"""
4-Class MLP Catastrophic Forgetting Demonstration

This script demonstrates catastrophic forgetting by training an MLP on two tasks:
1. First task: Learn to classify classes 0 and 1
2. Second task: Learn to classify classes 2 and 3

The network will forget how to classify the first task when learning the second task.
"""

import math
import random


class MLP4ClassClassifier:
    # Class constants
    INPUT_SIZE = 2
    NUM_CLASSES = 4
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_HIDDEN_SIZE = 8
    BIAS_INIT_RANGE = 0.05
    WEIGHT_INIT_RANGE = 0.1
    EPSILON = 1e-15
    Z_CLAMP_MIN = -500
    Z_CLAMP_MAX = 500
    WEIGHT_SAVE_INTERVAL = 5
    PRINT_INTERVAL = 10
    
    def __init__(self, learning_rate=DEFAULT_LEARNING_RATE, hidden_size=DEFAULT_HIDDEN_SIZE):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_classes = self.NUM_CLASSES
        
        # Initialize weights with small random values
        self.weights1 = self._initialize_weights(self.INPUT_SIZE, hidden_size)  # Input to hidden
        self.bias1 = [random.uniform(-self.BIAS_INIT_RANGE, self.BIAS_INIT_RANGE) for _ in range(hidden_size)]
        
        self.weights2 = self._initialize_weights(hidden_size, self.num_classes)  # Hidden to output
        self.bias2 = [random.uniform(-self.BIAS_INIT_RANGE, self.BIAS_INIT_RANGE) for _ in range(self.num_classes)]
        
        self.epoch = 0
        self.loss_history = []
        self.accuracy_history = []
        self.weight_history = []
        
        # Store initial weights
        self._save_weight_snapshot()
    
    def _initialize_weights(self, input_size, output_size):
        """Initialize weights with small random values"""
        weights = []
        for i in range(input_size):
            weights.append([random.uniform(-self.WEIGHT_INIT_RANGE, self.WEIGHT_INIT_RANGE) for _ in range(output_size)])
        return weights
    
    def _save_weight_snapshot(self):
        """Save current weights for history tracking"""
        self.weight_history.append({
            'epoch': self.epoch,
            'weights1': [row[:] for row in self.weights1],  # Deep copy
            'bias1': self.bias1[:],
            'weights2': [row[:] for row in self.weights2],  # Deep copy
            'bias2': self.bias2[:]
        })
    
    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        z = max(self.Z_CLAMP_MIN, min(self.Z_CLAMP_MAX, z))  # Clamp to prevent overflow
        return 1 / (1 + math.exp(-z))
    
    def _softmax(self, logits):
        """Softmax activation for output layer"""
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        return [exp / sum_exp for exp in exp_logits]
    
    def forward(self, x):
        """Forward pass through the network"""
        # Input to hidden layer
        hidden = []
        for j in range(self.hidden_size):
            sum_val = self.bias1[j]
            for i in range(len(x)):
                sum_val += x[i] * self.weights1[i][j]
            hidden.append(self._sigmoid(sum_val))
        
        # Hidden to output layer (logits)
        logits = []
        for k in range(self.num_classes):
            sum_val = self.bias2[k]
            for j in range(self.hidden_size):
                sum_val += hidden[j] * self.weights2[j][k]
            logits.append(sum_val)
        
        output = self._softmax(logits)
        
        return {'hidden': hidden, 'logits': logits, 'output': output}
    
    def predict(self, x):
        """Get prediction probabilities for input x"""
        return self.forward(x)['output']
    
    def predict_class(self, x):
        """Get predicted class for input x"""
        predictions = self.predict(x)
        return predictions.index(max(predictions))
    
    def train_step(self, dataset):
        """Perform one training step on the dataset"""
        total_loss = 0
        correct = 0
        
        for x, y in dataset:
            # Forward pass
            forward_result = self.forward(x)
            hidden = forward_result['hidden']
            output = forward_result['output']
            
            # Calculate cross-entropy loss
            loss = -math.log(output[y] + self.EPSILON)  # Add small epsilon to prevent log(0)
            total_loss += loss
            
            if self.predict_class(x) == y:
                correct += 1
            
            # Backward pass
            output_errors = output[:]
            output_errors[y] -= 1  # Derivative of cross-entropy + softmax
            
            # Calculate hidden layer errors
            hidden_errors = []
            for j in range(self.hidden_size):
                error = 0
                for k in range(self.num_classes):
                    error += output_errors[k] * self.weights2[j][k]
                hidden_errors.append(error * hidden[j] * (1 - hidden[j]))  # Sigmoid derivative
            
            # Update output layer weights
            for j in range(self.hidden_size):
                for k in range(self.num_classes):
                    self.weights2[j][k] -= self.learning_rate * output_errors[k] * hidden[j]
            for k in range(self.num_classes):
                self.bias2[k] -= self.learning_rate * output_errors[k]
            
            # Update hidden layer weights
            for i in range(len(x)):
                for j in range(self.hidden_size):
                    self.weights1[i][j] -= self.learning_rate * hidden_errors[j] * x[i]
            for j in range(self.hidden_size):
                self.bias1[j] -= self.learning_rate * hidden_errors[j]
        
        self.epoch += 1
        avg_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)
        
        self.loss_history.append(avg_loss)
        self.accuracy_history.append(accuracy)
        
        # Save weight snapshot every few epochs
        if self.epoch % self.WEIGHT_SAVE_INTERVAL == 0:
            self._save_weight_snapshot()
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, dataset, epochs=50, show_progress=True):
        """Train the model for specified epochs"""
        metrics = {}
        for epoch in range(epochs):
            metrics = self.train_step(dataset)
            if show_progress and (epoch + 1) % self.PRINT_INTERVAL == 0:
                print(f"Epoch {self.epoch}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.1%}")
        return metrics
    
    def evaluate(self, dataset):
        """Evaluate the model on a dataset"""
        correct = 0
        total_loss = 0
        
        for x, y in dataset:
            predictions = self.predict(x)
            predicted_class = self.predict_class(x)
            
            if predicted_class == y:
                correct += 1
            
            loss = -math.log(predictions[y] + self.EPSILON)
            total_loss += loss
        
        accuracy = correct / len(dataset)
        avg_loss = total_loss / len(dataset)
        
        return {'accuracy': accuracy, 'loss': avg_loss}
    
    def get_weight_magnitudes(self):
        """Calculate L2 norm of weight matrices"""
        w1_mag = 0
        for i in range(len(self.weights1)):
            for j in range(len(self.weights1[i])):
                w1_mag += self.weights1[i][j] ** 2
        w1_mag = math.sqrt(w1_mag)
        
        w2_mag = 0
        for j in range(len(self.weights2)):
            for k in range(len(self.weights2[j])):
                w2_mag += self.weights2[j][k] ** 2
        w2_mag = math.sqrt(w2_mag)
        
        return {'hidden': w1_mag, 'output': w2_mag}
    
    def reset(self):
        """Reset the network to initial random weights"""
        self.weights1 = self._initialize_weights(self.INPUT_SIZE, self.hidden_size)
        self.bias1 = [random.uniform(-self.BIAS_INIT_RANGE, self.BIAS_INIT_RANGE) for _ in range(self.hidden_size)]
        self.weights2 = self._initialize_weights(self.hidden_size, self.num_classes)
        self.bias2 = [random.uniform(-self.BIAS_INIT_RANGE, self.BIAS_INIT_RANGE) for _ in range(self.num_classes)]
        
        self.epoch = 0
        self.loss_history = []
        self.accuracy_history = []
        self.weight_history = []
        
        self._save_weight_snapshot()


def generate_task1_dataset():
    """Task 1: Classes 0 and 1 (first two classes)"""
    return [
        # Class 0 (red) - top-left quadrant
        ([1.5, 6.5], 0), ([2, 7], 0), ([1, 6], 0), ([2.5, 6.5], 0),
        ([1.5, 7.5], 0), ([2, 6], 0), ([1, 7], 0), ([2.5, 7.5], 0),
        
        # Class 1 (green) - bottom-right quadrant  
        ([6, 1.5], 1), ([7, 2], 1), ([6.5, 1], 1), ([7.5, 2.5], 1),
        ([6.5, 2.5], 1), ([7, 1.5], 1), ([6, 2], 1), ([7.5, 1.5], 1)
    ]


def generate_task2_dataset():
    """Task 2: Classes 2 and 3 (second two classes)"""
    return [
        # Class 2 (blue) - top-right quadrant
        ([6, 6.5], 2), ([7, 7], 2), ([6.5, 6], 2), ([7.5, 6.5], 2),
        ([6.5, 7.5], 2), ([7, 6], 2), ([6, 7], 2), ([7.5, 7.5], 2),
        
        # Class 3 (yellow) - bottom-left quadrant
        ([1.5, 1.5], 3), ([2, 2], 3), ([1, 1], 3), ([2.5, 2.5], 3),
        ([1.5, 2.5], 3), ([2, 1], 3), ([1, 2], 3), ([2.5, 1.5], 3)
    ]


def generate_all_classes_dataset():
    """Combined dataset for comparison"""
    return generate_task1_dataset() + generate_task2_dataset()


def get_class_name(class_id):
    """Get human-readable class name"""
    names = ['Red', 'Green', 'Blue', 'Yellow']
    return names[class_id]


def print_stats(metrics, weight_mags, phase="Training"):
    """Print training statistics"""
    print(f"{phase} Stats:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Hidden weights magnitude: {weight_mags['hidden']:.3f}")
    print(f"  Output weights magnitude: {weight_mags['output']:.3f}")
    print()


def test_task_knowledge(model, task_dataset, task_name):
    """Test model's knowledge on a specific task"""
    print(f"\n=== Testing {task_name} Knowledge ===")
    metrics = model.evaluate(task_dataset)
    weight_mags = model.get_weight_magnitudes()
    print_stats(metrics, weight_mags, "Test")
    
    # Test each class individually
    classes_in_task = set(y for _, y in task_dataset)
    for class_id in sorted(classes_in_task):
        class_samples = [(x, y) for x, y in task_dataset if y == class_id]
        class_metrics = model.evaluate(class_samples)
        print(f"  {get_class_name(class_id)} (Class {class_id}): {class_metrics['accuracy']:.1%}")
    
    return metrics


def main():
    """Main function demonstrating catastrophic forgetting"""
    print("=" * 60)
    print("4-Class MLP Catastrophic Forgetting Demonstration")
    print("=" * 60)
    print()
    
    print("This demonstrates catastrophic forgetting in neural networks.")
    print("We'll train on two sequential tasks:")
    print("  Task 1: Learn to classify Red (Class 0) and Green (Class 1)")
    print("  Task 2: Learn to classify Blue (Class 2) and Yellow (Class 3)")
    print()
    print("Watch how the network forgets Task 1 when learning Task 2!")
    print()
    
    # Initialize model and datasets
    model = MLP4ClassClassifier(learning_rate=MLP4ClassClassifier.DEFAULT_LEARNING_RATE, hidden_size=MLP4ClassClassifier.DEFAULT_HIDDEN_SIZE)
    task1_data = generate_task1_dataset()
    task2_data = generate_task2_dataset()
    all_data = generate_all_classes_dataset()
    
    # Phase 1: Train on Task 1 (Classes 0 and 1)
    print("=" * 50)
    print("PHASE 1: Training on Task 1 (Red and Green)")
    print("=" * 50)
    print(f"Training on {len(task1_data)} samples from classes 0 and 1...")
    print()
    
    # Train on task 1
    TASK_EPOCHS = 50
    PROGRESS_INTERVAL = 20
    for epoch in range(TASK_EPOCHS):
        metrics = model.train_step(task1_data)
        if (epoch + 1) % PROGRESS_INTERVAL == 0:
            weight_mags = model.get_weight_magnitudes()
            print(f"Epoch {model.epoch}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.1%}, "
                  f"Hidden Mag = {weight_mags['hidden']:.3f}, Output Mag = {weight_mags['output']:.3f}")
    
    print()
    print("Task 1 training complete!")
    
    # Test Task 1 knowledge
    task1_metrics_after_task1 = test_task_knowledge(model, task1_data, "Task 1")
    
    # Pause and ask user to continue
    print("\n" + "=" * 50)
    print("Ready to train on Task 2...")
    print("=" * 50)
    response = input("Press Enter to continue training on Task 2 (Classes 2 and 3), or 'q' to quit: ")
    if response.lower() == 'q':
        return
    
    # Phase 2: Train on Task 2 (Classes 2 and 3) 
    print("\n" + "=" * 50)
    print("PHASE 2: Training on Task 2 (Blue and Yellow)")
    print("=" * 50)
    print(f"Training on {len(task2_data)} samples from classes 2 and 3...")
    print("This will cause catastrophic forgetting of Task 1!")
    print()
    
    # Train on task 2
    for epoch in range(TASK_EPOCHS):
        metrics = model.train_step(task2_data)
        if (epoch + 1) % PROGRESS_INTERVAL == 0:
            weight_mags = model.get_weight_magnitudes()
            print(f"Epoch {model.epoch}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.1%}, "
                  f"Hidden Mag = {weight_mags['hidden']:.3f}, Output Mag = {weight_mags['output']:.3f}")
    
    print()
    print("Task 2 training complete!")
    
    # Test Task 2 knowledge
    task2_metrics = test_task_knowledge(model, task2_data, "Task 2")
    
    # Pause and ask user to continue to testing
    print("\n" + "=" * 50)
    print("Ready to test catastrophic forgetting...")
    print("=" * 50)
    response = input("Press Enter to test how well the model remembers Task 1, or 'q' to quit: ")
    if response.lower() == 'q':
        return
    
    # Phase 3: Test catastrophic forgetting
    print("\n" + "=" * 60)
    print("PHASE 3: Testing Catastrophic Forgetting")
    print("=" * 60)
    
    # Test Task 1 knowledge after Task 2 training
    print("Testing Task 1 knowledge AFTER Task 2 training:")
    task1_metrics_after_task2 = test_task_knowledge(model, task1_data, "Task 1 (After Task 2)")
    
    # Summary
    print("\n" + "=" * 60)
    print("CATASTROPHIC FORGETTING SUMMARY")
    print("=" * 60)
    print(f"Task 1 accuracy BEFORE Task 2: {task1_metrics_after_task1['accuracy']:.1%}")
    print(f"Task 1 accuracy AFTER Task 2:  {task1_metrics_after_task2['accuracy']:.1%}")
    print(f"Task 2 accuracy:                {task2_metrics['accuracy']:.1%}")
    print()
    
    accuracy_drop = task1_metrics_after_task1['accuracy'] - task1_metrics_after_task2['accuracy']
    print(f"Catastrophic forgetting: {accuracy_drop:.1%} drop in Task 1 performance!")
    
    SEVERE_FORGETTING_THRESHOLD = 0.3  # 30% drop
    MODERATE_FORGETTING_THRESHOLD = 0.1  # 10% drop
    
    if accuracy_drop > SEVERE_FORGETTING_THRESHOLD:
        print("🔴 Severe catastrophic forgetting observed!")
    elif accuracy_drop > MODERATE_FORGETTING_THRESHOLD:
        print("🟡 Moderate catastrophic forgetting observed.")
    else:
        print("🟢 Minimal forgetting - network retained most Task 1 knowledge.")
    
    print()
    print("Final test on all classes:")
    final_metrics = test_task_knowledge(model, all_data, "All Classes")


if __name__ == "__main__":
    main()