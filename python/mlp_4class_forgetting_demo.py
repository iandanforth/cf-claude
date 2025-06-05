#!/usr/bin/env python3
"""
Non-interactive demo of 4-Class MLP Catastrophic Forgetting

This version runs automatically without user input for testing purposes.
"""

from mlp_4class_forgetting import (
    MLP4ClassClassifier, 
    generate_task1_dataset, 
    generate_task2_dataset, 
    generate_all_classes_dataset,
    get_class_name,
    print_stats,
    test_task_knowledge
)


def main():
    """Non-interactive demonstration of catastrophic forgetting"""
    print("=" * 60)
    print("4-Class MLP Catastrophic Forgetting Demo (Non-Interactive)")
    print("=" * 60)
    print()
    
    print("This demonstrates catastrophic forgetting in neural networks.")
    print("Training on two sequential tasks:")
    print("  Task 1: Learn to classify Red (Class 0) and Green (Class 1)")
    print("  Task 2: Learn to classify Blue (Class 2) and Yellow (Class 3)")
    print()
    
    # Initialize model and datasets
    model = MLP4ClassClassifier(learning_rate=MLP4ClassClassifier.DEFAULT_LEARNING_RATE, hidden_size=MLP4ClassClassifier.DEFAULT_HIDDEN_SIZE)
    task1_data = generate_task1_dataset()
    task2_data = generate_task2_dataset()
    all_data = generate_all_classes_dataset()
    
    # Phase 1: Train on Task 1
    print("=" * 50)
    print("PHASE 1: Training on Task 1 (Red and Green)")
    print("=" * 50)
    print(f"Training on {len(task1_data)} samples from classes 0 and 1...")
    print()
    
    TASK_EPOCHS = 100
    PROGRESS_INTERVAL = 20
    
    for epoch in range(TASK_EPOCHS):
        metrics = model.train_step(task1_data)
        if (epoch + 1) % PROGRESS_INTERVAL == 0:
            weight_mags = model.get_weight_magnitudes()
            print(f"Epoch {model.epoch}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.1%}, "
                  f"Hidden Mag = {weight_mags['hidden']:.3f}, Output Mag = {weight_mags['output']:.3f}")
    
    print("\nTask 1 training complete!")
    task1_metrics_after_task1 = test_task_knowledge(model, task1_data, "Task 1")
    
    # Phase 2: Train on Task 2
    print("\n" + "=" * 50)
    print("PHASE 2: Training on Task 2 (Blue and Yellow)")
    print("=" * 50)
    print(f"Training on {len(task2_data)} samples from classes 2 and 3...")
    print("This will cause catastrophic forgetting of Task 1!")
    print()
    
    for epoch in range(TASK_EPOCHS):
        metrics = model.train_step(task2_data)
        if (epoch + 1) % PROGRESS_INTERVAL == 0:
            weight_mags = model.get_weight_magnitudes()
            print(f"Epoch {model.epoch}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.1%}, "
                  f"Hidden Mag = {weight_mags['hidden']:.3f}, Output Mag = {weight_mags['output']:.3f}")
    
    print("\nTask 2 training complete!")
    task2_metrics = test_task_knowledge(model, task2_data, "Task 2")
    
    # Phase 3: Test catastrophic forgetting
    print("\n" + "=" * 60)
    print("PHASE 3: Testing Catastrophic Forgetting")
    print("=" * 60)
    
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
    
    print("\nFinal test on all classes:")
    final_metrics = test_task_knowledge(model, all_data, "All Classes")


if __name__ == "__main__":
    main()