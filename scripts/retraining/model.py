import torch.nn as nn

class SiameseNetwork(nn.Module):

    def __init__(self, embedding_dimensionality, dropout, number_neurons, number_layers):
        super(SiameseNetwork, self).__init__()

        # define dropout layer in __init__ to disable it with model.eval()
        self.dropout_layer = nn.Dropout(dropout)

        layers = []
        last_output_dim = embedding_dimensionality
        # Setting up the Fully Connected Layers
        for i in range(number_layers):
            linear_layer = nn.Linear(last_output_dim, number_neurons)
            layers.append(linear_layer)
            layers.append(nn.ReLU(inplace=True))
            layers.append(self.dropout_layer)

            last_output_dim = number_neurons

        self.layers = nn.ModuleList(layers)
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2