from sklearn.metrics import f1_score
import numpy as np


# Example data
# Ground truth labels (true values)
y_true = [
    [1, 0, 0, 1, 0, 0],  # For id 00001cee341fdb12
    [1, 0, 1, 0, 1, 1],  # For id 0000247867823ef7
    [0, 0, 1, 0, 0, 0],  # For id 00013b17ad220c46
    [1, 1, 1, 0, 0, 0],  # For id 00017563c3f7919a
    [1, 0, 0, 1, 0, 1]   # For id 00017695ad8997eb
    # ....
]

# Predicted labels
y_pred = [
    [1, 0, 0, 1, 0, 0],  # Predicted for id 00001cee341fdb12
    [1, 0, 1, 0, 1, 0],  # Predicted for id 0000247867823ef7
    [0, 0, 1, 0, 0, 0],  # Predicted for id 00013b17ad220c46
    [1, 1, 1, 0, 0, 0],  # Predicted for id 00017563c3f7919a
    [1, 0, 0, 1, 0, 1]   # Predicted for id 00017695ad8997eb
    # ....
]

# Calculate the Macro F1 Score
macro_f1 = f1_score(y_true, y_pred, average='macro')

print(f"Macro F1 Score: {macro_f1:.4f}")