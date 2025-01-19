import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Custom Dataset class
class SeizureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
train_set_df = pd.read_csv('C:\\college_stuff\\events\\impulse\\task6.1\\train_set_df.csv')

# Prepare data for training
X = train_set_df.drop(columns=["class"])
X = X.fillna(0)  # Handle NaNs
y = train_set_df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

# Create DataLoader objects
train_dataset = SeizureDataset(X_train, y_train)
test_dataset = SeizureDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class SeizureClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SeizureClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Model, loss function, and optimizer
input_size = X.shape[1]
num_classes = len(np.unique(y))

model = SeizureClassifier(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        # Move data to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for inputs, labels in test_loader:
        # Move data to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_predictions.extend(predictions.cpu().numpy())  # Move to CPU for metrics
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_predictions)
balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
roc_auc = roc_auc_score(
    all_labels, np.array(all_probabilities), multi_class="ovr"
)

print(f"Accuracy Score: {accuracy:.4f}")
print(f"Balanced Accuracy Score: {balanced_acc:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

