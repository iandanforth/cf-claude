#!/usr/bin/env python3
"""
Basic tests for the MLP 4-class classifier
"""

from mlp_4class_forgetting import (
    MLP4ClassClassifier,
    generate_task1_dataset,
    generate_task2_dataset,
    generate_all_classes_dataset
)


def test_model_initialization():
    """Test that model initializes correctly"""
    model = MLP4ClassClassifier()
    assert model.learning_rate == MLP4ClassClassifier.DEFAULT_LEARNING_RATE
    assert model.hidden_size == MLP4ClassClassifier.DEFAULT_HIDDEN_SIZE
    assert model.num_classes == MLP4ClassClassifier.NUM_CLASSES
    assert len(model.weights1) == MLP4ClassClassifier.INPUT_SIZE  # 2 inputs
    assert len(model.weights1[0]) == MLP4ClassClassifier.DEFAULT_HIDDEN_SIZE  # 8 hidden units
    assert len(model.weights2) == MLP4ClassClassifier.DEFAULT_HIDDEN_SIZE  # 8 hidden units
    assert len(model.weights2[0]) == MLP4ClassClassifier.NUM_CLASSES  # 4 output classes
    print("✓ Model initialization test passed")


def test_forward_pass():
    """Test forward pass functionality"""
    model = MLP4ClassClassifier()
    x = [1.0, 2.0]
    
    result = model.forward(x)
    assert 'hidden' in result
    assert 'logits' in result 
    assert 'output' in result
    assert len(result['hidden']) == MLP4ClassClassifier.DEFAULT_HIDDEN_SIZE
    assert len(result['logits']) == MLP4ClassClassifier.NUM_CLASSES
    assert len(result['output']) == MLP4ClassClassifier.NUM_CLASSES
    
    # Test softmax properties
    SOFTMAX_TOLERANCE = 1e-10
    EXPECTED_SOFTMAX_SUM = 1.0
    output_sum = sum(result['output'])
    assert abs(output_sum - EXPECTED_SOFTMAX_SUM) < SOFTMAX_TOLERANCE, f"Softmax sum should be {EXPECTED_SOFTMAX_SUM}, got {output_sum}"
    assert all(p >= 0 for p in result['output']), "All probabilities should be non-negative"
    
    print("✓ Forward pass test passed")


def test_prediction():
    """Test prediction functions"""
    model = MLP4ClassClassifier()
    x = [1.0, 2.0]
    
    predictions = model.predict(x)
    predicted_class = model.predict_class(x)
    
    assert len(predictions) == MLP4ClassClassifier.NUM_CLASSES
    assert 0 <= predicted_class < MLP4ClassClassifier.NUM_CLASSES
    assert predicted_class == predictions.index(max(predictions))
    
    print("✓ Prediction test passed")


def test_datasets():
    """Test dataset generation"""
    task1_data = generate_task1_dataset()
    task2_data = generate_task2_dataset()
    all_data = generate_all_classes_dataset()
    
    TASK1_SIZE = 16
    TASK2_SIZE = 16
    ALL_DATA_SIZE = 32
    assert len(task1_data) == TASK1_SIZE
    assert len(task2_data) == TASK2_SIZE
    assert len(all_data) == ALL_DATA_SIZE
    
    # Check task 1 has only classes 0 and 1
    task1_classes = set(y for _, y in task1_data)
    assert task1_classes == {0, 1}
    
    # Check task 2 has only classes 2 and 3
    task2_classes = set(y for _, y in task2_data)
    assert task2_classes == {2, 3}
    
    # Check all data has all classes
    all_classes = set(y for _, y in all_data)
    assert all_classes == {0, 1, 2, 3}
    
    print("✓ Dataset generation test passed")


def test_training():
    """Test training functionality"""
    model = MLP4ClassClassifier()
    task1_data = generate_task1_dataset()
    
    initial_epoch = model.epoch
    initial_loss_history_len = len(model.loss_history)
    
    # Train for a few epochs
    TEST_EPOCHS = 5
    model.train(task1_data, epochs=TEST_EPOCHS, show_progress=False)
    
    assert model.epoch == initial_epoch + TEST_EPOCHS
    assert len(model.loss_history) == initial_loss_history_len + TEST_EPOCHS
    assert len(model.accuracy_history) == len(model.loss_history)
    
    print("✓ Training test passed")


def test_evaluation():
    """Test evaluation functionality"""
    model = MLP4ClassClassifier()
    task1_data = generate_task1_dataset()
    
    metrics = model.evaluate(task1_data)
    assert 'accuracy' in metrics
    assert 'loss' in metrics
    MIN_ACCURACY = 0.0
    MAX_ACCURACY = 1.0
    MIN_LOSS = 0.0
    assert MIN_ACCURACY <= metrics['accuracy'] <= MAX_ACCURACY
    assert metrics['loss'] >= MIN_LOSS
    
    print("✓ Evaluation test passed")


def test_weight_magnitudes():
    """Test weight magnitude calculation"""
    model = MLP4ClassClassifier()
    mags = model.get_weight_magnitudes()
    
    assert 'hidden' in mags
    assert 'output' in mags
    MIN_MAGNITUDE = 0
    assert mags['hidden'] >= MIN_MAGNITUDE
    assert mags['output'] >= MIN_MAGNITUDE
    
    print("✓ Weight magnitude test passed")


def run_all_tests():
    """Run all tests"""
    print("Running MLP 4-class classifier tests...")
    print()
    
    test_model_initialization()
    test_forward_pass()
    test_prediction()
    test_datasets()
    test_training()
    test_evaluation()
    test_weight_magnitudes()
    
    print()
    print("🎉 All tests passed!")


if __name__ == "__main__":
    run_all_tests()