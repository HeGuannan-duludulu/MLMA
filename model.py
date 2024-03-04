import torch.nn as nn
import resnet


class resnet_extractor(nn.Module):
    def __init__(self, input_channel, dropout=None, num_layers=2):
        super(resnet_extractor, self).__init__()

        self.model = resnet.ResNet(input_channel, resnet.ResidualBlock, num_layers, dropout)

        self.out_features = self.model.out_features

    def forward(self, input):
        output = self.model(input)
        return output


class ImprovedABNClassifier(nn.Module):
    def __init__(self, in_features, nnClassCount, dropout=0.5):
        """
        Improved classifier based on fully connected layers with additional features for better performance.
        :param in_features: Number of input features.
        :param nnClassCount: Number of classes for classification.
        :param dropout: Dropout rate for regularization.
        """
        super(ImprovedABNClassifier, self).__init__()

        # Layer definitions
        self.fc1 = nn.Linear(in_features, 512)  # First fully connected layer
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization for the first layer
        self.relu = nn.ReLU()  # ReLU activation
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer
        self.bn2 = nn.BatchNorm1d(256)  # Batch normalization for the second layer
        self.fc3 = nn.Linear(256, nnClassCount)  # Final fully connected layer for classification

        # Initialize weights using Xavier initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input data.
        :return: Output after passing through the layers.
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        # x = nn.functional.sigmoid(x)
        return x
