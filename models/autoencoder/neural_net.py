import torch as T
import torch.nn as nn
device = T.device("cuda" if T.cuda.is_available() else "cpu")


class NeuralNet(nn.Module):

    def __init__(self, input_dimension: int, output_dimension: int, hidden_dimension: int, encoded_dimension: int, num_hidden_encoder: int = 1, num_hidden_decoder: int = 1, drop_probability: float = 0, activation: str = "relu"):
        super().__init__()

        if hidden_dimension <= encoded_dimension:
            hidden_dimension = encoded_dimension * 2

        self.encoder_input = nn.Linear(
            in_features=input_dimension, out_features=hidden_dimension)
        
        self.encoder_hidden = nn.ModuleList()
        
        
        c = hidden_dimension - encoded_dimension
        for _ in range(num_hidden_encoder - 1):
            self.encoder_hidden.append(
                nn.Linear(in_features=encoded_dimension + c, out_features=encoded_dimension + c//2))
            c //= 2

        self.encoder_output = nn.Linear(
            in_features=encoded_dimension + c, out_features=encoded_dimension
        )
        
        
        c = hidden_dimension - encoded_dimension

        self.dropout = nn.Dropout(drop_probability)

        self.decoder_input = nn.Linear(
            in_features=encoded_dimension, out_features= encoded_dimension + c//(2**(num_hidden_decoder - 1)),
        )
        
        self.decoder_hidden = nn.ModuleList()
        for i in range(num_hidden_decoder - 1):
            self.decoder_hidden.append(
                nn.Linear(in_features=encoded_dimension + c//(2**(num_hidden_decoder - 1 - i)),
                          out_features=encoded_dimension + c//(2**(num_hidden_decoder - 2 - i)))
            )

        self.decoder_output = nn.Linear(
            in_features=hidden_dimension, out_features=output_dimension
        )

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "selu":
            self.activation = nn.SELU()
        else:
            self.activation = nn.ELU()

        self.dropout_factor = 1
        
    def set_training(self):
        self.dropout_factor = 1
        
    def set_inference(self):
        self.dropout_factor = 1
        
    def forward(self, x):
        #encoder
        x = self.encoder_input(x)
        x = self.activation(x)

        for layer in self.encoder_hidden:
            x = layer(x)
            x = self.activation(x)

        x = self.encoder_output(x)
        x = self.activation(x)
        
        if self.dropout_factor > 0:
            x = self.dropout(x)
        #decoder
        x = self.decoder_input(x)
        x = self.activation(x)
        
        for layer in self.decoder_hidden:
            x = layer(x)
            x = self.activation(x)
            
        x = self.decoder_output(x)
        x = self.activation(x)
        
        return x
        
        