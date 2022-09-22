import torch.nn as nn

class SiameseNetwork(nn.Module):

    def __init__(self, embedding_dimensionality, dropout, number_neurons, number_layers):
        super(SiameseNetwork, self).__init__()

        # define dropout layer in __init__ to disable it with model.eval()
        if dropout:
            self.dropout_layer = nn.Dropout(dropout)

        layers = []
        last_output_dim = embedding_dimensionality
        for i in range(number_layers):
            linear_layer = nn.Linear(last_output_dim, number_neurons)
            layers.append(linear_layer)
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(self.dropout_layer)
            last_output_dim = number_neurons

        self.layers = nn.ModuleList(layers)
        
    def forward_once(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2