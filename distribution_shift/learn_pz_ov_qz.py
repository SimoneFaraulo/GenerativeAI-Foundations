import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

# --- 1. Define Distribution Parameters ---
# p(x) is our target distribution
mean_p = 0.0
std_p = 1.0
# q(x) is our proposal/source distribution
mean_q = 1.0
std_q = 1.0
n_train_samples = 5000  # Number of training samples to draw from each distribution
n_test_samples = 1000   # A more robust number of test samples for accuracy evaluation

# --- 2. Create Distribution Objects ---
# This is the distribution for the class labeled '1'
norm_p = torch.distributions.Normal(mean_p, std_p)
# This is the distribution for the class labeled '0'
norm_q = torch.distributions.Normal(mean_q, std_q)

# --- 3. Generate Training and Test Data ---
# Generate samples from q(x) and label them as class 0
x_train_q = norm_q.sample((n_train_samples, ))
y_train_q = torch.zeros(n_train_samples)
x_test_q = norm_q.sample((n_test_samples,))
y_test_q = torch.zeros(n_test_samples)

# Generate samples from p(x) and label them as class 1
x_train_p = norm_p.sample((n_train_samples, ))
y_train_p = torch.ones(n_train_samples)
x_test_p = norm_p.sample((n_test_samples,))
y_test_p = torch.ones(n_test_samples)

# --- 4. Assemble and Shuffle the Datasets ---
# Concatenate the samples from both distributions to create the full datasets.
X_train = torch.cat((x_train_q, x_train_p))
Y_train = torch.cat((y_train_q, y_train_p))
X_test = torch.cat((x_test_q, x_test_p))
Y_test = torch.cat((y_test_q, y_test_p))

# Reshape data for scikit-learn (which expects 2D arrays) and shuffle the training set.
# Shuffling is good practice to ensure the model sees data in a random, unbiased order.
X_train_reshaped = X_train.reshape(-1, 1)
X_train_shuffled, Y_train_shuffled = shuffle(X_train_reshaped, Y_train, random_state=42)

# --- 5. Train and Evaluate the Classifier ---
# Define a more capable MLP classifier. The original `hidden_layer_sizes=2` was too small.
classifier = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', random_state=1, max_iter=1000)

# Train the classifier on the shuffled training data
print("Training the classifier...")
classifier.fit(X_train_shuffled, Y_train_shuffled)
print("Training complete.")

# Evaluate the classifier's accuracy on the test set
score = classifier.score(X_test.reshape(-1, 1), Y_test)
print(f"\nClassifier accuracy on the test set: {score:.2%}")

# --- 6. Visualize the Learned Probability ---
# Now we verify that the classifier has learned P(b=1 | x) = p(x) / (p(x) + q(x)).

# Create a grid of x-values for plotting.
x_grid = torch.linspace(-5, 5, 500).reshape(-1, 1)

# Use the trained classifier to predict the probability of class 1 for each point on the grid.
# .predict_proba(X) returns probabilities for [class 0, class 1]. We select the second column.
prob_b1_given_x_learned = classifier.predict_proba(x_grid)[:, 1]

# Calculate the true, theoretical probability P(b=1 | x).
p_x_on_grid = torch.exp(norm_p.log_prob(x_grid.squeeze()))
q_x_on_grid = torch.exp(norm_q.log_prob(x_grid.squeeze()))
prob_b1_given_x_theoretical = p_x_on_grid / (p_x_on_grid + q_x_on_grid)

# --- Plot the comparison ---
plt.figure(figsize=(12, 7))
plt.title("Classifier's Learned Probability vs. Theoretical Probability")

# Plot the probability learned by the classifier.
plt.plot(x_grid.numpy(), prob_b1_given_x_learned, label='Learned D(x) = P(b=1|x)', color='red', linewidth=3)

# Plot the theoretical probability for comparison.
plt.plot(x_grid.numpy(), prob_b1_given_x_theoretical.numpy(), label='Theoretical p(x)/(p(x)+q(x))', color='blue', linestyle='--', linewidth=2)

# Also plot the original distributions for context.
plt.plot(x_grid.numpy(), p_x_on_grid.numpy(), label='p(x)', color='black', linestyle=':', alpha=0.5)
plt.plot(x_grid.numpy(), q_x_on_grid.numpy(), label='q(x)', color='gray', linestyle=':', alpha=0.5)

plt.xlabel("x")
plt.ylabel("Probability / Density")
plt.legend()
plt.grid(True)
plt.show()