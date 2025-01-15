import sys
sys.path.append('d:/итмо/нир/applyBN')

import pytest
import numpy as np
from applybn.feature_selection.ce_feature_selector import CausalFeatureSelector  

# Test 1: Check the correctness of data discretization
def test_discretize_data_iqr():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    selector = CausalFeatureSelector(data=data, target=None)
    discretized_data, bins = selector.discretize_data_iqr(data)
    
    # Ensure all discretized values are within the bin indices range
    assert np.all(discretized_data >= 0)
    assert np.all(discretized_data < len(bins) - 1)

# Test 2: Check the correctness of entropy calculation
def test_entropy():
    data = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    selector = CausalFeatureSelector(data=data, target=None)
    discretized_data, _ = selector.discretize_data_iqr(data)
    entropy_value = selector.entropy(discretized_data)
    
    # Entropy should never be negative
    assert entropy_value >= 0

# Test 3: Check the correctness of conditional entropy calculation
def test_conditional_entropy():
    X = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    Y = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    
    selector = CausalFeatureSelector(data=X.reshape(-1, 1), target=Y)
    X_discretized = selector.discretize_data_iqr(X)[0]
    Y_discretized = selector.discretize_data_iqr(Y)[0]
    
    cond_entropy_value = selector.conditional_entropy(X_discretized, Y_discretized)
    
    # Conditional entropy should never be negative
    assert cond_entropy_value >= 0

# Test 4: Check the correctness of causal effect calculation
def test_causal_effect():
    X = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    Y = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    other_features = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1], [0, 0], [0, 1]])

    selector = CausalFeatureSelector(data=X.reshape(-1, 1), target=Y)
    causal_effect_value = selector.causal_effect(X, Y, other_features)
    
    # Ensure causal effect returns a float
    assert isinstance(causal_effect_value, float)

# Test 5: Check the correctness of feature selection
def test_causal_feature_selection():
    data = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0], [0, 0, 2], [1, 1, 1], [2, 2, 2]])
    target = np.array([0, 1, 0, 1, 1, 0])
    
    selector = CausalFeatureSelector(data=data, target=target)
    selected_features = selector.causal_feature_selection(data, target)
    
    # Ensure the function returns a list of feature indices
    assert isinstance(selected_features, list)
    assert all(isinstance(i, int) for i in selected_features)

# Test 6: Check for error handling with mismatched sizes
def test_inconsistent_sizes():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    target = np.array([1, 2])  # Target size does not match the data size
    
    selector = CausalFeatureSelector(data=data, target=None)
    
    # Ensure a ValueError is raised for mismatched dimensions
    with pytest.raises(ValueError):
        selector.causal_feature_selection(data, target)

