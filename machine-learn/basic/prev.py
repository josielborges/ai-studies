from sklearn.svm import LinearSVC

"""
This code trains a LinearSVC model with 6 compounds and their corresponding labels (soluble or not soluble).
The compounds are represented as lists of 3 elements, which represent the presence or absence of certain chemical groups.
The labels are represented as strings, either "S" for soluble or "N" for not soluble.
The model is then used to predict the solubility of 3 new compounds.
The predictions are printed out, along with a translation of the labels to human-readable strings.
"""

compound1 = [1, 1, 1]
compound2 = [0, 0, 0]
compound3 = [1, 0, 1]
compound4 = [0, 1, 0]
compound5 = [1, 1, 0]
compound6 = [0, 0, 1]

# The compounds and their labels
train_data = [compound1, compound2, compound3, compound4, compound5, compound6]
train_labels = ["S", "N", "S", "N", "S", "S"]

# Train the model
model = LinearSVC()
model.fit(train_data, train_labels)

test1 = [1, 0, 0]
test2 = [0, 1, 1]
test3 = [1, 0, 1]

# The new compounds to be tested
test_data = [test1, test2, test3]

# Get the predictions
predictions = model.predict(test_data)

# Map the labels to human-readable strings
predictions_map = {
    "S": "Soluble",
    "N": "Not soluble"
}

print('Predictions od the model to tested compounds:', predictions)

# Print the predictions with the labels translated
for i, prediction in enumerate(predictions):
    print(f'Compound {i + 1} could be considered: {predictions_map[prediction]}')
