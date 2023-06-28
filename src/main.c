#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

// TODO: add safeguards for matrix sizes etc

typedef enum {
    Mean_Squared_Error,
    Binary_Cross_Entropy
} Loss_Function_Type;

typedef enum {
    Sigmoid,
    ReLU
} Activation_Function_Type;

typedef struct {
    float* activations;
    int num_activations;
    float* weighted_inputs;
    int num_weighted_inputs;
} Cross_Layer_Data;

typedef struct {
    int num_layers;
    int* layer_sizes;
    float* weights;
    float* biases;
    int num_weights;
    int num_biases;
    Cross_Layer_Data*  cross_layer_data;
    float* weight_gradients;
    float* bias_gradients;
} Neural_Network;


int sum_cross_layer_data_activations_until_layer(Neural_Network* neural_network, int layer) {
    int total_num_activations = 0;
    for (int i = 0; i < layer; i++) {
        int num_rows = neural_network->layer_sizes[i];

        for (int j = 0; j < num_rows; j++) {
            total_num_activations++;
        }
    }
    return total_num_activations;
}

int sum_cross_layer_data_weighted_inputs_until_layer(Neural_Network* neural_network, int layer) {
    int total_num_weighted_inputs = 0;
    for (int i = 0; i < layer; i++) {
        int num_rows = neural_network->layer_sizes[i];

        for (int j = 0; j < num_rows; j++) {
            total_num_weighted_inputs++;
        }
    }
    return total_num_weighted_inputs;
}

float* neural_network_cross_layer_data_activations_at(Neural_Network* neural_network, int layer, int index) {
    return &neural_network->cross_layer_data->activations[sum_cross_layer_data_activations_until_layer(neural_network, layer) + index];
}

float* neural_network_cross_layer_data_weighted_inputs_at(Neural_Network* neural_network, int layer, int index) {
    return &neural_network->cross_layer_data->weighted_inputs[sum_cross_layer_data_weighted_inputs_until_layer(neural_network, layer) + index];
}

float sigmoid(float x) {
    return 1.0f / (1.0f + (float)exp(-(double)x));
}

float sigmoid_derivative(float x) {
    return sigmoid(x) * (1.0f - sigmoid(x));
}

float relu(float x) {
    return (float)fmax(0, x);
}

float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

float activation_function(float x, Activation_Function_Type activation_function_type) {
    float activation = 0.0f;

    if (activation_function_type == Sigmoid) {
        activation = sigmoid(x);
    }
    else if (activation_function_type == ReLU) {
        activation = relu(x);
    }

    return activation;
}

float activation_function_derivative(float x, Activation_Function_Type activation_function_type) {
    float activation_derivative = 0.0f;

    if (activation_function_type == Sigmoid) {
        activation_derivative = sigmoid_derivative(x);
    }
    else if (activation_function_type == ReLU) {
        activation_derivative = relu_derivative(x);
    }

    return activation_derivative;
}

int sum_weights_until_layer(Neural_Network* neural_network, int layer) {
    int total_num_weights = 0;

    // NOTE: index of weight matrix - layer index
    for (int i = 1; i < layer; i++) {
        int num_rows = neural_network->layer_sizes[i];

        // NOTE: row in the matrix
        for (int j = 0; j < num_rows; j++) {
            int num_columns = neural_network->layer_sizes[i-1];

            // NOTE: column in the matrix
            for (int k = 0; k < num_columns; k++) {
                total_num_weights++;
            }
        }
    }
    return total_num_weights;
}

int sum_biases_until_layer(Neural_Network* neural_network, int layer) {
    int total_num_biases = 0;
    // NOTE: index of bias vector - layer index
    for (int i = 1; i < layer; i++) {
        int num_rows = neural_network->layer_sizes[i];

        // NOTE: row in the bias vector
        for (int j = 0; j < num_rows; j++) {
            total_num_biases++;
        }
    }
    return total_num_biases;
}

float* neural_network_weight_at(Neural_Network* neural_network, int layer, int row, int column) {
    return &neural_network->weights[sum_weights_until_layer(neural_network, layer) + row *  neural_network->layer_sizes[layer - 1] + column];
}

float* neural_network_bias_at(Neural_Network* neural_network, int layer, int row) {
    return &neural_network->biases[sum_biases_until_layer(neural_network, layer) + row];
}

float* neural_network_weight_gradient_at(Neural_Network* neural_network, int layer, int row, int column) {
    return &neural_network->weight_gradients[sum_weights_until_layer(neural_network, layer) + row *  neural_network->layer_sizes[layer - 1] + column];
}

float* neural_network_bias_gradient_at(Neural_Network* neural_network, int layer, int row) {
    return &neural_network->bias_gradients[sum_biases_until_layer(neural_network, layer) + row];
}

void destroy_neural_network(Neural_Network* neural_network) {
    free((void*)neural_network->layer_sizes);
    free((void*)neural_network->weights);
    free((void*)neural_network->biases);
    free((void*)neural_network->weight_gradients);
    free((void*)neural_network->bias_gradients);
    free((void*)neural_network->cross_layer_data->activations);
    free((void*)neural_network->cross_layer_data->weighted_inputs);
    free((void*)neural_network->cross_layer_data);
    free((void*)neural_network);
}

Neural_Network* initiate_neural_network(int num_layers, int* layer_sizes) {
    // NOTE: initiate the struct
    Neural_Network* neural_network = (Neural_Network*) malloc(sizeof(Neural_Network));

    // NOTE: fill in all of the base variables
    // NOTE: layer data
    neural_network->num_layers = num_layers;
    neural_network->layer_sizes = malloc(sizeof(int) * num_layers);
    memcpy(neural_network->layer_sizes, layer_sizes, num_layers * sizeof(int));

    // NOTE: initiate weights and biases
    int total_size_weights = sum_weights_until_layer(neural_network, num_layers);
    int total_size_biases = sum_biases_until_layer(neural_network, num_layers);
    neural_network->weights = (float*)malloc(total_size_weights * sizeof(float));
    neural_network->biases = (float*)malloc(total_size_biases * sizeof(float));
    neural_network->weight_gradients = (float*)malloc(total_size_weights * sizeof(float));
    neural_network->bias_gradients = (float*)malloc(total_size_biases * sizeof(float));

    // NOTE: fill in weights and biases
    // NOTE: the weight data is structured the following way:
    // NOTE: {  [ (...), (...), ... (...) ], [ (...), (...), ... (...) ], ... [ (...), (...), ... (...) ]  }
    // NOTE: the bias data is structured the following way:
    // NOTE: {  [ ... ], [ ... ], ... [ ... ]  }

    // NOTE: fill in values for the weights and biases
    for (int i = 1; i < neural_network->num_layers; i++) {
        // NOTE: weight matrix size at i: neural_network->layer_sizes[i] x neural_network->layer_sizes[i - 1]
        // NOTE: bias vector size at i: neural_network->layer_sizes[i] x 1
        int num_rows = neural_network->layer_sizes[i];
        for (int j = 0; j < num_rows; j++) {
            int num_columns = neural_network->layer_sizes[i - 1];
            int bias_index = sum_biases_until_layer(neural_network, i) + j;
            // TODO: initiate with better starting biases
//            neural_network->biases[bias_index] = 0.1f * (float)bias_index;
            *neural_network_bias_at(neural_network, i, j) = 0.1f * (float)bias_index;

            // NOTE: column in the matrix
            for (int k = 0; k < num_columns; k++) {
                int weight_index = sum_weights_until_layer(neural_network, i) + j * num_columns + k;
                // TODO: initiate with better starting weights
                *neural_network_weight_at(neural_network, i, j, k) = 0.1f * (float)weight_index;
//                neural_network->weights[weight_index] = 0.1f * (float)weight_index;
            }
        }
    }

    // NOTE: cross layer data, serves to save: activations, weighted inputs and ... across multiple layers TODO whatever i will need add to list
    neural_network->cross_layer_data = (Cross_Layer_Data*)malloc(sizeof(Cross_Layer_Data));
    neural_network->cross_layer_data->num_activations = 0;
    neural_network->cross_layer_data->num_weighted_inputs = 0;

    // TODO: add a NOTE
    // TODO: figure out why this is and what this describes
    for (int i = 0; i < neural_network->num_layers; i++) {
        for (int x = 0; x < neural_network->layer_sizes[i]; x++) {
            neural_network->cross_layer_data->num_weighted_inputs++;
            neural_network->cross_layer_data->num_activations++;
        }
    }

    // TODO: add a NOTE
    // TODO: figure out the sizes of these and where t find them
    // NOTE: initiate the cross layer data
    neural_network->cross_layer_data->activations = (float*)malloc(neural_network->cross_layer_data->num_activations * sizeof(float));
    neural_network->cross_layer_data->weighted_inputs = (float*)malloc(neural_network->cross_layer_data->num_weighted_inputs * sizeof(float));

    // NOTE: fill all the cross layer data
    // TODO: better values
    int n = 0;
    for (int i = 0; i < neural_network->num_layers; i++) {
        for (int x = 0; x < neural_network->layer_sizes[i]; x++) {
            *neural_network_cross_layer_data_weighted_inputs_at(neural_network, i, x) = 0.1f * (float)n;
            *neural_network_cross_layer_data_activations_at(neural_network, i, x) = activation_function(0.1f * (float)n, Sigmoid);
            n++;
        }
    }

    // NOTE: finally fill out these two for easier use
    neural_network->num_weights = sum_weights_until_layer(neural_network, neural_network->num_layers - 1);
    neural_network->num_biases = sum_biases_until_layer(neural_network, neural_network->num_layers - 1);

    return neural_network;
}

void evaluate_neural_network(Neural_Network* neural_network, const float* inputs, float* outputs) {
    // NOTE: feed through the inputs
    // TODO: finish the cross layer stuff, make it be able to access activations and inputs from other layer to be able to access it from line 209 lague: 34:23

    // NOTE: initiate the modular layer inputs
    float* layer_inputs = (float*)malloc(sizeof(float) * neural_network->layer_sizes[0]);
    // NOTE: fill in the layer inputs from the parameters
    for (int x = 0; x < neural_network->layer_sizes[0]; x++) {
        layer_inputs[x] = inputs[x];
    }

    // NOTE: loop through all the layers
    for (int i = 1; i < neural_network->num_layers; i++) {

        // NOTE: neural_network->layer_sizes[i] is the layer output size and the rows of the weight matrix
        // NOTE: neural_network->layer_sizes[i - 1] is the input layer size and the columns of the weight matrix

        // NOTE: initiate layer_outputs to the size of the current layer
        float* layer_outputs = (float*)malloc(sizeof(float) * neural_network->layer_sizes[i]);
        // NOTE: fill the layer outputs with bias values since they would be added onto the sum anyway
        for (int x = 0; x < neural_network->layer_sizes[i]; x++) {
            float bias_i_x = *neural_network_bias_at(neural_network, i, x);
            layer_outputs[x] = bias_i_x;
        }

        // NOTE: loop through the weight matrix's rows
        for (int x = 0; x < neural_network->layer_sizes[i]; x++) {
            // NOTE: loop through the weight matrix's columns
            for (int y = 0; y < neural_network->layer_sizes[i - 1]; y++) {
                layer_outputs[x] += layer_inputs[y] * *neural_network_weight_at(neural_network, i, x, y);
            }

            // NOTE: save the weighted inputs into the cross layer data (TODO: check)
            *neural_network_cross_layer_data_weighted_inputs_at(neural_network, i, x) = layer_outputs[x];
            // NOTE: use the activation function on the layer outputs
            layer_outputs[x] = activation_function(layer_outputs[x], Sigmoid);
            // NOTE: save the activations into the cross layer data (TODO: check)
            *neural_network_cross_layer_data_activations_at(neural_network, i, x) = layer_outputs[x];
        }

        // NOTE: free the allocated layer inputs
        free((void*)layer_inputs);
        // NOTE: reallocate
        layer_inputs = (float*)malloc(sizeof(float) * neural_network->layer_sizes[i]);
        // NOTE: make the inputs for the next layer the current outputs
        for (int x = 0; x < neural_network->layer_sizes[i]; x++){
            layer_inputs[x] = layer_outputs[x];
        }

        // NOTE: if we are on the last layer, just make the outputs passed in the arguments the current-layer outputs
        if (i == neural_network->num_layers - 1) {
            for (int x = 0; x < neural_network->layer_sizes[neural_network->num_layers - 1]; x++) {
                outputs[x] = layer_outputs[x];
            }
        }

        // NOTE: free the current-layer outputs
        free((void*)layer_outputs);
    }
    free((void*)layer_inputs);
}

float mean_squared_error_loss(const float* output, const float* desired_output, float* node_errors, int output_size) {
    // NOTE: MSE = (1/n) * Σ(y_pred - y_true)^2
    float loss = 0.0f;
    
    for (int i = 0; i < output_size; i++) {
        if (node_errors)
            node_errors[i] = (output[i] - desired_output[i]) * (output[i] - desired_output[i]);
        loss += (output[i] - desired_output[i]) * (output[i] - desired_output[i]);
    }
    loss /= (float)output_size;

    return loss;
}

float binary_cross_entropy_loss(float* output, float* desired_output, float* node_errors, int output_size) {
    // NOTE: BCE = - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    // TODO: implement this one
    return 0.0f;
}

float calculate_loss(float* output, float* desired_output, float* node_errors, int output_size, Loss_Function_Type loss_function_type) {
    float loss = 0.0f;

    if (loss_function_type == Mean_Squared_Error) {
        loss = mean_squared_error_loss(output, desired_output, node_errors, output_size);
    }
    else if (loss_function_type == Binary_Cross_Entropy) {
        loss = binary_cross_entropy_loss(output, desired_output, node_errors, output_size);
    }

    return loss;
}

float mean_squared_error_loss_derivative(const float* output, const float* desired_output, float* node_errors_derivative, int output_size) {
    // NOTE: MSE' = (1/n) * Σ (2 * (y_pred - y_true))
    float loss_derivative = 0.0f;

    for (int i = 0; i < output_size; i++)
    {
        node_errors_derivative[i] = 2.0f * (output[i] - desired_output[i]);
        loss_derivative += 2.0f * (output[i] - desired_output[i]);
    }
    loss_derivative /= (float)output_size;

    return loss_derivative;
}

float binary_cross_entropy_loss_derivative(float* output, float* desired_output, float* node_errors_derivative, int output_size) {
    // NOTE: BCE' = TODO
    // TODO: implement this one
    return 0.0f;
}

float calculate_loss_derivative(float* output, float* desired_output, float* node_errors_derivative, int output_size, Loss_Function_Type loss_function_type) {
    float loss_derivative = 0.0f;

    if (loss_function_type == Mean_Squared_Error) {
        loss_derivative = mean_squared_error_loss_derivative(output, desired_output, node_errors_derivative, output_size);
    }
    else if (loss_function_type == Binary_Cross_Entropy) {
        loss_derivative = binary_cross_entropy_loss_derivative(output, desired_output, node_errors_derivative, output_size);
    }

    return loss_derivative;
}

void train_neural_network(Neural_Network* neural_network, float* input, float* desired_output, float learning_rate) {
    printf("\n[LOG] Training started\n");
    // NOTE: initiate the output values from the evaluate() function
    float* output_values = (float*)malloc(neural_network->layer_sizes[neural_network->num_layers - 1] * sizeof(float));
    // NOTE: retrieve them from the evaluation
    evaluate_neural_network(neural_network, input, output_values);
    float cost = calculate_loss(output_values, desired_output, NULL, neural_network->layer_sizes[neural_network->num_layers - 1], Mean_Squared_Error);
    printf("[LOG] Cost = %f\n", cost);

    // NOTE: the output layer (extra)
    {
        int output_layer_size = neural_network->layer_sizes[neural_network->num_layers - 1];
        // NOTE: calculate output layer node values
        float *cost_derivative = calculate_loss_derivative();
    }


    // NOTE: update gradients
    for (int l = neural_network->num_layers - 1; l >= 1; l--){

        /* for (int x = 0; x < neural_network->layer_sizes[l]; x++) {
            printf("[LOG] Backpropagation: Layer %d: Weighted Inputs = %f\n", l, *neural_network_cross_layer_data_weighted_inputs_at(neural_network, l, x));
            printf("[LOG] Backpropagation: Layer %d: Activations = %f\n", l, *neural_network_cross_layer_data_activations_at(neural_network, l, x));
        } */
//        printf("[LOG] Backpropagation: Layer %d: Loss = %f\n", l, loss);


        // NOTE: fill in random values to the weight and bias gradients TODO real values
        for (int i = 1; i < neural_network->num_layers; i++) {
            int num_rows = neural_network->layer_sizes[i];
            for (int j = 0; j < num_rows; j++) {
                int num_columns = neural_network->layer_sizes[i - 1];
                *neural_network_bias_gradient_at(neural_network, i, j) = (float)(rand() % 100) / 100.0f;

                // NOTE: column in the matrix
                for (int k = 0; k < num_columns; k++) {
                    *neural_network_weight_gradient_at(neural_network, i, j, k) = (float)(rand() % 100) / 100.0f;
                }
            }
        }

        // NOTE: free what was allocated
//        free((void*)node_errors);
    }

    // NOTE: apply gradients
    {
        for (int i = 1; i < neural_network->num_layers; i++) {
            int num_rows = neural_network->layer_sizes[i];
            for (int j = 0; j < num_rows; j++) {
                int num_columns = neural_network->layer_sizes[i - 1];
                *neural_network_bias_at(neural_network, i, j) -= learning_rate * *neural_network_bias_gradient_at(neural_network, i, j);

                // NOTE: column in the matrix
                for (int k = 0; k < num_columns; k++) {
                    *neural_network_weight_at(neural_network, i, j, k) -= learning_rate * *neural_network_weight_gradient_at(neural_network, i, j, k);
                }
            }
        }

    }

    // NOTE: free the output from evaluate()
    free((void*)output_values);

    printf("[LOG] Training complete\n\n");
}

int main()
{
    int  layer_sizes[3] = {2, 3, 1};
    Neural_Network* neural_network = initiate_neural_network(sizeof(layer_sizes) / sizeof(int), layer_sizes);

    // ------------------------

//    float outputs[1] = {0.0f};
//    float inputs[2] = {1.0f, 0.0f};
//    evaluate_neural_network(neural_network, inputs, outputs);
//    printf("%f\n", outputs[0]);

    float training_inputs[2] = {0.0f, 0.0f};
    float training_outputs[1] = {0.0f};
    for (int i = 0; i < 5; i++)
        train_neural_network(neural_network, training_inputs, training_outputs, 0.5f);

    // ------------------------

    destroy_neural_network(neural_network);

    return 0;
}
