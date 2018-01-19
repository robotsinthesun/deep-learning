import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # Define sigmoid as activation function.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
                    

    def train(self, features, targets):

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        #print("Delta weights input to hidden:")
        #print(delta_weights_i_h)
        #print("Delta weights hidden to output:")
        #print(delta_weights_h_o)
        
        # Loop through all training samples.
        for X, y in zip(features, targets):
            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass(X)  
            
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

        
        

    def forward_pass(self, X):

        # Calc hidden layer outputs from inputs and weights.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calc final output from hidden layer values and weights.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs # Use f(x) = x instead of sigmoid.
       
        return final_outputs, hidden_outputs

    
    
    
    def backpropagation(self, final_outputs, hidden_outputs, X, y, deltaWeightsInputToHidden, deltaWeightsHiddenToOutput):

        # Define output error derivative using square error measure E = 1/2 (yHat - y)^2.
        # Use a factor of 1/2 that will nicely lead to dE = (y - final_outputs).
        dErrorOut = (y - final_outputs)
        #print("Error derivative at the output layer.")
        #print(dErrorOut)
        
        # These are the error's partial derivatives corresponding to the weights between hidden and output layer.
        dErrorHiddenToOutput = dErrorOut * hidden_outputs
        #print("Error gradient between output and hidden layer: ")
        #print(dErrorHiddenToOutput)
        
        #print("Weights between output and hidden layer: ")
        #print(self.weights_hidden_to_output.reshape(self.weights_hidden_to_output.shape[0],))
        
        # These are the errors at the hidden layer outputs.
        #errorHidden = dErrorHiddenToOutput * self.weights_hidden_to_output.reshape(self.weights_hidden_to_output.shape[0],)
        errorHidden = dErrorOut * self.weights_hidden_to_output.reshape(self.weights_hidden_to_output.shape[0],)
        #print("Error at the hidden layer.")
        #print(errorHidden)
        
        # These are the error's partial derivatives corresponding to the weights between input and hidden layer.        
        dErrorInputToHidden = np.dot(X[:,None], (errorHidden * hidden_outputs * (1 - hidden_outputs))[:,None].T) 
        #print("Error gradient between hidden and input layer: ")
        #print(dErrorInputToHidden)
        
        # Sum up the error gradients at the weights.
        # These will be used to update the weights.
        # Don't forget to divide these guys by the sample size to get the average!
        # Input to hidden.
        deltaWeightsInputToHidden += dErrorInputToHidden
        # Hidden to output.
        deltaWeightsHiddenToOutput += dErrorHiddenToOutput[:,None]
        
        return deltaWeightsInputToHidden, deltaWeightsHiddenToOutput

    
    
    
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        # Average the error gradients and apply the learning rate.
        self.weights_hidden_to_output += self.lr * (delta_weights_h_o / float(n_records))
        self.weights_input_to_hidden += self.lr * (delta_weights_i_h / float(n_records))

        
        
        
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        # Do the forward pass.
        final_outputs = self.forward_pass(features)[0]
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 10000
learning_rate = 1.
#hidden_nodes = 4
output_nodes = 1
