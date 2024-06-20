import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


# preprocessed data and labels
preprocessed_data = pd.read_csv("preprocessed_BTCGBP_1000_4.csv", header=None)
X_train = preprocessed_data.iloc[:, 0].values
y_train = preprocessed_data.iloc[:, 1].values

print("Train data shape:", X_train.shape)
print("Train labels shape:", y_train.shape)

# print("Example pre-processed data:\n", preprocessed_data.head(20))
# print(train_data[:10])
# print(train_labels[:10])



# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        output = self.sigmoid(output)
        return output

# hyperparameters
input_size = 4 # number of features per tick: (X)tick_id, price, volume, time, type
hidden_size = 50
output_size = 1
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# the LSTM model
model = LSTMModel(input_size, hidden_size, output_size)

# loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training data to torch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train).unsqueeze(1)

# create a dataLoader for batch training
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# training
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# evaluation
with torch.no_grad():
    outputs = model(X_train_tensor)
    binary_predictions = (outputs >= 0.5).float()

# training accuracy
accuracy = (binary_predictions == y_train_tensor).float().mean().item() * 100
print("Training Accuracy: {:.2f}%".format(accuracy))
