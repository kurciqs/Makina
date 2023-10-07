#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define EPSILON 0.01f
#define EXIT(x) {printf(x); exit(-69);}

#define DETAILED_DEBUG

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
    float* weight_gradients;
    float* bias_gradients;
    int num_weights;
    int num_biases;
    Cross_Layer_Data*  cross_layer_data;
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
    if (layer < 0 || layer >= neural_network->num_layers)
        EXIT("[ERROR] Accessed non existent activation!")
    return &neural_network->cross_layer_data->activations[sum_cross_layer_data_activations_until_layer(neural_network, layer) + index];
}

float* neural_network_cross_layer_data_weighted_inputs_at(Neural_Network* neural_network, int layer, int index) {
    if (layer == 0 || layer >= neural_network->num_layers)
        EXIT("[ERROR] Accessed non existent weighted input!")
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
    if (layer == 0 || layer >= neural_network->num_layers)
        EXIT("[ERROR] Accessed non existent weights!")
    return &neural_network->weights[sum_weights_until_layer(neural_network, layer) + row *  neural_network->layer_sizes[layer - 1] + column];
}

float* neural_network_bias_at(Neural_Network* neural_network, int layer, int row) {
    if (layer == 0 || layer >= neural_network->num_layers)
        EXIT("[ERROR] Accessed non existent biases!")
    return &neural_network->biases[sum_biases_until_layer(neural_network, layer) + row];
}

float* neural_network_weight_gradient_at(Neural_Network* neural_network, int layer, int row, int column) {
    if (layer == 0 || layer >= neural_network->num_layers)
        EXIT("[ERROR] Accessed non existent weight gradient!")
    return &neural_network->weight_gradients[sum_weights_until_layer(neural_network, layer) + row * neural_network->layer_sizes[layer - 1] + column];
}

float* neural_network_bias_gradient_at(Neural_Network* neural_network, int layer, int row) {
    if (layer == 0 || layer >= neural_network->num_layers)
        EXIT("[ERROR] Accessed non existent bias gradient!")
    return &neural_network->bias_gradients[sum_biases_until_layer(neural_network, layer) + row];
}

void destroy_neural_network(Neural_Network* neural_network) {
    free((void*)neural_network->layer_sizes);
    free((void*)neural_network->weights);
    free((void*)neural_network->biases);
    free((void*)neural_network->cross_layer_data->activations);
    free((void*)neural_network->cross_layer_data->weighted_inputs);
    free((void*)neural_network->cross_layer_data);
    free((void*)neural_network->bias_gradients);
    free((void*)neural_network->weight_gradients);
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

    // NOTE fan_avg is the average of the fan-in and fan-out (input and output units) of the layer.
    float fan_avg = 2.0f;

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
            *neural_network_bias_at(neural_network, i, j) = 0.0f;
            *neural_network_bias_gradient_at(neural_network, i, j) = 0.0f;

            // NOTE: column in the matrix
            for (int k = 0; k < num_columns; k++) {
                // NOTE: W = np.random.randn(shape) * np.sqrt(1.0 / fan_avg)
                *neural_network_weight_at(neural_network, i, j, k) = (float)((rand() % 100) / 100.0f) * sqrtf(1.0f / fan_avg);
                *neural_network_weight_gradient_at(neural_network, i, j, k) = 0.0f;
            }
        }
    }

    // NOTE: cross layer data, serves to save: activations, weighted inputs and ... across multiple layers
    neural_network->cross_layer_data = (Cross_Layer_Data*)malloc(sizeof(Cross_Layer_Data));
    neural_network->cross_layer_data->num_activations = 0;
    neural_network->cross_layer_data->num_weighted_inputs = 0;

    // NOTE: gives access to weighted inputs and activations across layers
    for (int i = 0; i < neural_network->num_layers; i++) {
        for (int x = 0; x < neural_network->layer_sizes[i]; x++) {
            neural_network->cross_layer_data->num_weighted_inputs++;
            neural_network->cross_layer_data->num_activations++;
        }
    }

    // NOTE: initiate the cross layer data
    neural_network->cross_layer_data->activations = (float*)malloc(neural_network->cross_layer_data->num_activations * sizeof(float));
    neural_network->cross_layer_data->weighted_inputs = (float*)malloc(neural_network->cross_layer_data->num_weighted_inputs * sizeof(float));

    // NOTE: fill all the cross layer data
    int n = 0;
    for (int i = 0; i < neural_network->num_layers; i++) {
        for (int x = 0; x < neural_network->layer_sizes[i]; x++) {
            if (i)
                *neural_network_cross_layer_data_weighted_inputs_at(neural_network, i, x) = 0.0f;
            *neural_network_cross_layer_data_activations_at(neural_network, i, x) = 0.0f;
            n++;
        }
    }

    // NOTE: finally fill out these two for easier use
    neural_network->num_weights = sum_weights_until_layer(neural_network, neural_network->num_layers);
    neural_network->num_biases = sum_biases_until_layer(neural_network, neural_network->num_layers);

    return neural_network;
}

void evaluate_neural_network(Neural_Network* neural_network, const float* inputs, float* outputs, int mode) {
    // NOTE: feed through the inputs
#ifndef DETAILED_DEBUG
    printf("[LOG] Evaluation:\n");
#endif
    // NOTE: initiate the modular layer inputs
    float* layer_inputs = (float*)malloc(sizeof(float) * neural_network->layer_sizes[0]);
    // NOTE: fill in the layer inputs from the parameters
    for (int x = 0; x < neural_network->layer_sizes[0]; x++) {
        layer_inputs[x] = inputs[x];
#ifndef DETAILED_DEBUG
        printf("Evaluation input at row %d = %f\n", x, layer_inputs[x]);
#endif
        if (mode == 0)
            *neural_network_cross_layer_data_activations_at(neural_network, 0, x) = layer_inputs[x];
    }

    // NOTE: mode determines whether to update cross-layer data
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
#ifndef DETAILED_DEBUG
            printf("bias at layer %d, row %d = %f\n", i, x, bias_i_x);
#endif
        }

        // NOTE: loop through the weight matrix's rows
        for (int x = 0; x < neural_network->layer_sizes[i]; x++) {
            // NOTE: loop through the weight matrix's columns
#ifndef DETAILED_DEBUG
            printf("[LOG] Node %d\n", x);
#endif
            for (int y = 0; y < neural_network->layer_sizes[i - 1]; y++) {
#ifndef DETAILED_DEBUG
                printf("weighted input at layer %d, row %d = %f\n", i, x, layer_outputs[x]);
#endif
                float weight_i_x_y = *neural_network_weight_at(neural_network, i, x, y);
#ifndef DETAILED_DEBUG
                printf("weight at layer %d, row %d, column %d = %f\n", i, x, y, weight_i_x_y);
#endif
                layer_outputs[x] += layer_inputs[y] * weight_i_x_y;
#ifndef DETAILED_DEBUG
                printf("weighted input at layer %d, row %d, with respect to layer input %d += (%f * %f = %f)\n", i, x, y, layer_inputs[y], weight_i_x_y, layer_inputs[y] * weight_i_x_y);
#endif
            }

            // NOTE: save the weighted inputs into the cross layer data
            if (mode == 0)
                *neural_network_cross_layer_data_weighted_inputs_at(neural_network, i, x) = layer_outputs[x];
#ifndef DETAILED_DEBUG
            printf("final weighted input at layer %d, row %d = %f\n", i, x, layer_outputs[x]);
#endif

            // NOTE: use the activation function on the layer outputs
            layer_outputs[x] = activation_function(layer_outputs[x], Sigmoid);
            // NOTE: save the activations into the cross layer data
            if (mode == 0)
                *neural_network_cross_layer_data_activations_at(neural_network, i, x) = layer_outputs[x];
#ifndef DETAILED_DEBUG
            printf("activation at layer %d, row %d = %f\n", i, x, layer_outputs[x]);
#endif
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
        if (node_errors_derivative)
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

void compute_gradients_neural_network(Neural_Network* neural_network, float* input, float* desired_output, int number_total_training_examples) {
#ifndef DETAILED_DEBUG
    printf("\n[LOG] Updating gradients for ONE training example started\n");
#endif
    // NOTE: initiate the output values from the evaluate() function
    float* output_values = (float*)malloc(neural_network->layer_sizes[neural_network->num_layers - 1] * sizeof(float));
    evaluate_neural_network(neural_network, input, output_values, 0);

    // NOTE: essentially the number of activations since every neuron except for the input neurons has technically its own bias
    float* cost_derivative_activations = (float*)malloc(sum_biases_until_layer(neural_network, neural_network->num_layers) * sizeof(float));
    memset(cost_derivative_activations, 0, sum_biases_until_layer(neural_network, neural_network->num_layers) * sizeof(float));

    //NOTE: the output layer (extra)
    {
        int output_layer_size = neural_network->layer_sizes[neural_network->num_layers - 1];

        // NOTE: get the change in the cost with respect to the activations of the last layer (predicted output)
        float* cost_derivative_nodes = (float*)malloc(output_layer_size * sizeof(float));
        __attribute__((unused)) float cost_derivative = calculate_loss_derivative(output_values, desired_output, cost_derivative_nodes, output_layer_size, Mean_Squared_Error);

        // NOTE: loop through every output node or row in the weight matrix of the last layer
        for (int j = 0; j < output_layer_size; j++) {

            // NOTE: this is the change in the cost with respect to the activation at node j, so 2*(y_hat - y)
            cost_derivative_activations[sum_biases_until_layer(neural_network, neural_network->num_layers) - 1 - j] = cost_derivative_nodes[j];

            // NOTE: 1/m * 1 * f'(z_j) * dC/da_j
             *neural_network_bias_gradient_at(neural_network, neural_network->num_layers - 1, j) +=
                     (1.0f/(float)number_total_training_examples)
                     *
                     1.0f
                     *
                     activation_function_derivative(*neural_network_cross_layer_data_weighted_inputs_at(neural_network, neural_network->num_layers - 1, j), Sigmoid)
                     *
                     cost_derivative_activations[sum_biases_until_layer(neural_network, neural_network->num_layers) - 1 - j];

            // NOTE: loop through every node in the layer before the last layer or column in the weight matrix of the last layer
            for (int k = 0; k < neural_network->layer_sizes[neural_network->num_layers - 2]; k++) {
                // NOTE: 1/m * a_k^(l-1) * f'(z_j) * dC/da_j
                *neural_network_weight_gradient_at(neural_network, neural_network->num_layers - 1, j, k) +=
                        (1.0f/(float)number_total_training_examples)
                        *
                        *neural_network_cross_layer_data_activations_at(neural_network, neural_network->num_layers - 2, k)
                        *
                        activation_function_derivative(*neural_network_cross_layer_data_weighted_inputs_at(neural_network, neural_network->num_layers - 1, j), Sigmoid)
                        *
                        cost_derivative_activations[sum_biases_until_layer(neural_network, neural_network->num_layers) - 1 - j];
            }
        }


        free((void*)cost_derivative_nodes);
    }
#ifndef DETAILED_DEBUG
    printf("[LOG] Gradients:\n");
#endif
    // NOTE: the other layers
    for (int l = neural_network->num_layers - 2; l > 0; l--){
        for (int j = 0; j < neural_network->layer_sizes[l]; j++) {
            float cda_local = 0.0f;
            for (int p = 0; p < neural_network->layer_sizes[l+1]; p++) {
                cda_local +=
                        *neural_network_weight_at(neural_network, l+1, p, j)
                        *
                        activation_function_derivative(*neural_network_cross_layer_data_weighted_inputs_at(neural_network, l+1, p), Sigmoid)
                        *
                        cost_derivative_activations[sum_biases_until_layer(neural_network, l+2) - 1 - p];
            }
            cost_derivative_activations[sum_biases_until_layer(neural_network, l+1) - 1 - j] += cda_local;
            *neural_network_bias_gradient_at(neural_network, l, j) =
                    (1.0f/(float)number_total_training_examples)
                    *
                    1.0f
                    *
                    activation_function_derivative(*neural_network_cross_layer_data_weighted_inputs_at(neural_network, l, j), Sigmoid)
                    *
                    cda_local;
#ifndef DETAILED_DEBUG
            printf("bias gradient at layer %d, row %d = %f\n", l, j, *neural_network_bias_gradient_at(neural_network, l, j));
#endif

            for (int k = 0; k < neural_network->layer_sizes[l - 1]; k++) {
                 *neural_network_weight_gradient_at(neural_network, l, j, k) +=
                        (1.0f/(float)number_total_training_examples)
                        *
                        *neural_network_cross_layer_data_activations_at(neural_network, l - 1, k)
                        *
                        activation_function_derivative(*neural_network_cross_layer_data_weighted_inputs_at(neural_network, l, j), Sigmoid)
                        *
                        cda_local;
#ifndef DETAILED_DEBUG
                printf("weight gradient at layer %d, row %d, column %d = %f\n", l, j, k, *neural_network_weight_gradient_at(neural_network, l, j, k));
#endif
            }
        }
    }

    // NOTE: free allocated memory
    free((void*)output_values);
    free((void*)cost_derivative_activations);

#ifndef DETAILED_DEBUG
    printf("[LOG] Updating gradients for ONE training example complete\n\n");
#endif
}

void apply_gradients_neural_network(Neural_Network* neural_network, float learning_rate) {
#ifndef DETAILED_DEBUG
    printf("\n[LOG] Applying gradients started\n");
#endif

    // NOTE: apply gradients
    {
        for (int i = 1; i < neural_network->num_layers; i++) {
            int num_rows = neural_network->layer_sizes[i];
            for (int j = 0; j < num_rows; j++) {
                int num_columns = neural_network->layer_sizes[i - 1];
                *neural_network_bias_at(neural_network, i, j) -= learning_rate * *neural_network_bias_gradient_at(neural_network, i, j);
                *neural_network_bias_gradient_at(neural_network, i, j) = 0.0f;

                // NOTE: column in the matrix
                for (int k = 0; k < num_columns; k++) {
                    *neural_network_weight_at(neural_network, i, j, k) -= learning_rate * *neural_network_weight_gradient_at(neural_network, i, j, k);
                    *neural_network_weight_gradient_at(neural_network, i, j, k) = 0.0f;
                }
            }
        }
    }
#ifndef DETAILED_DEBUG
    printf("[LOG] Applying gradients complete\n\n");
#endif
}

float neural_network_cost_for_one_training_example(Neural_Network* neural_network, float* input, float* desired_output) {
    // NOTE: calculate the overall cost for this one training example for debug purposes
    // NOTE: initiate the output values from the evaluate() function
    float* output_values = (float*)malloc(neural_network->layer_sizes[neural_network->num_layers - 1] * sizeof(float));
    // NOTE: retrieve them from the evaluation
    evaluate_neural_network(neural_network, input, output_values, 1);

    float cost = calculate_loss(output_values, desired_output, NULL, neural_network->layer_sizes[neural_network->num_layers - 1], Mean_Squared_Error);

    return cost;
}

float test_neural_network(Neural_Network* neural_network, const float* testing_data, int num_testing_samples) {
    // NOTE: returns essentially the average cost
    float total_cost = 0.0f;

    int input_size = neural_network->layer_sizes[0];
    int output_size = neural_network->layer_sizes[neural_network->num_layers - 1];
    float* inputs = (float*)malloc(input_size * sizeof(float));
    float* outputs = (float*)malloc(output_size * sizeof(float));

    // NOTE: which sample are we on
    for (int i = 0; i < num_testing_samples; i++) {
        // NOTE: which input of the sample are we on and we write it into an array
        for (int j = 0; j < input_size; j++) {
            inputs[j] = testing_data[i * (input_size + output_size) + j];
        }
        // NOTE: which output of the sample are we on and we write it into an array
        for (int j = 0; j < output_size; j++) {
            outputs[j] = testing_data[i * (input_size + output_size) + input_size + j];
        }
        total_cost += neural_network_cost_for_one_training_example(neural_network, inputs, outputs);
    }


    return total_cost / (float)num_testing_samples;
}

void neural_network_print_gradients(Neural_Network* neural_network) {
    for (int i = 1; i < neural_network->num_layers; i++) {
        int num_rows = neural_network->layer_sizes[i];
        for (int j = 0; j < num_rows; j++) {
            int num_columns = neural_network->layer_sizes[i - 1];
            printf("Bias gradient %d %d: %f\n", i, j, *neural_network_bias_gradient_at(neural_network, i, j));

            // NOTE: column in the matrix
            for (int k = 0; k < num_columns; k++) {
                printf("Weight gradient %d %d %d: %f\n", i, j, k, *neural_network_weight_gradient_at(neural_network, i, j, k));
            }
        }
    }
}

void neural_network_print_numeric_gradients(Neural_Network* neural_network, const float* training_data) {
    printf("\nNumeric Gradients:\n");
    Neural_Network* neural_network_2 = (Neural_Network*) malloc(sizeof(Neural_Network));
    memcpy(neural_network_2, neural_network, sizeof(Neural_Network));

    int input_size = neural_network_2->layer_sizes[0];
    int output_size = neural_network_2->layer_sizes[neural_network_2->num_layers - 1];
    float* inputs = (float*)malloc(input_size * sizeof(float));
    float* desired_outputs = (float*)malloc(output_size * sizeof(float));

    int training_example_index = 0;
    for (int s = 0; s < input_size; s++) {
        inputs[s] = training_data[training_example_index * (input_size + output_size) + s];
    }
    // NOTE: which output of the sample are we on and we write it into an array
    for (int t = 0; t < output_size; t++) {
        desired_outputs[t] = training_data[training_example_index * (input_size + output_size) + input_size + t];
    }

    for (int i = 1; i < neural_network_2->num_layers; i++) {
        int num_rows = neural_network_2->layer_sizes[i];
        for (int j = 0; j < num_rows; j++) {
            int num_columns = neural_network_2->layer_sizes[i - 1];
            // NOTE: we are checking for bias layer i index j

            // NOTE: change parameter and instantly revert the changes but get the cost in between
            *neural_network_bias_at(neural_network_2, i, j) -= EPSILON;
            float Jb0 = neural_network_cost_for_one_training_example(neural_network_2, inputs, desired_outputs);
            *neural_network_bias_at(neural_network_2, i, j) += EPSILON;

            // NOTE: change parameter and instantly revert the changes but get the cost in between
            *neural_network_bias_at(neural_network_2, i, j) += EPSILON;
            float Jb1 = neural_network_cost_for_one_training_example(neural_network_2, inputs, desired_outputs);
            *neural_network_bias_at(neural_network_2, i, j) -= EPSILON;

            *neural_network_bias_gradient_at(neural_network_2, i, j) = (Jb1 - Jb0) / (2.0f * EPSILON);
//            *neural_network_bias_gradient_at(neural_network, i, j) = *neural_network_bias_gradient_at(neural_network_2, i, j);

            printf("Bias gradient %d %d: %f\n", i, j, *neural_network_bias_gradient_at(neural_network_2, i, j));

            // NOTE: column in the matrix
            for (int k = 0; k < num_columns; k++) {
                // NOTE: we are checking for weight layer i row j column k

                // NOTE: change parameter and instantly revert the changes but get the cost in between
                *neural_network_weight_at(neural_network_2, i, j, k) -= EPSILON;
                float Jw0 = neural_network_cost_for_one_training_example(neural_network_2, inputs, desired_outputs);
                *neural_network_weight_at(neural_network_2, i, j, k) += EPSILON;

                // NOTE: change parameter and instantly revert the changes but get the cost in between
                *neural_network_weight_at(neural_network_2, i, j, k) += EPSILON;
                float Jw1 = neural_network_cost_for_one_training_example(neural_network_2, inputs, desired_outputs);
                *neural_network_weight_at(neural_network_2, i, j, k) -= EPSILON;

                *neural_network_weight_gradient_at(neural_network_2, i, j, k) = (Jw1 - Jw0) / (2.0f * EPSILON);

                printf("Weight gradient %d %d %d: %f\n", i, j, k, *neural_network_weight_gradient_at(neural_network_2, i, j, k));
//                *neural_network_weight_gradient_at(neural_network, i, j, k) = *neural_network_weight_gradient_at(neural_network_2, i, j, k);
            }
        }
    }

    free((void*)inputs);
    free((void*)desired_outputs);
    free((void*)neural_network_2);
}

void train_neural_network(Neural_Network* neural_network, float* training_data, int num_training_samples, float learning_rate, int training_epochs, int print_cost_every) {
    // NOTE: data is just flat array x0, ..., xn-1, y0, ym-1, x00, ..., x0n-1, ...
    clock_t t;
    t = clock();

    printf("\n[LOG] Training for %d iterations.\n", training_epochs);

    int input_size = neural_network->layer_sizes[0];
    int output_size = neural_network->layer_sizes[neural_network->num_layers - 1];
    float* inputs = (float*)malloc(input_size * sizeof(float));
    float* outputs = (float*)malloc(output_size * sizeof(float));

    for (int e = 0; e < training_epochs; e++) {
        int i = rand() % num_training_samples;
        // NOTE: which input of the sample are we on and we write it into an array
        for (int j = 0; j < input_size; j++) {
            inputs[j] = training_data[i * (input_size + output_size) + j];
        }
        // NOTE: which output of the sample are we on and we write it into an array
        for (int j = 0; j < output_size; j++) {
            outputs[j] = training_data[i * (input_size + output_size) + input_size + j];
        }

        compute_gradients_neural_network(neural_network, inputs, outputs, num_training_samples);
//        neural_network_print_gradients(neural_network);
//        neural_network_print_numeric_gradients(neural_network, training_data);
        apply_gradients_neural_network(neural_network, learning_rate);
        if (e % (print_cost_every) == 0) {
            float neural_network_cost = test_neural_network(neural_network, training_data, num_training_samples);
            printf("[LOG] Cost on training_data at iteration %d = %f\n", e, neural_network_cost);
        }
    }

    free((void*)inputs);
    free((void*)outputs);

    t = clock() - t;
    float time_taken = (float)(((double)t)/CLOCKS_PER_SEC);
    printf("[LOG] Training for %d iterations took %f seconds.\n\n", training_epochs, time_taken);
}

float* read_cvs_into_dataset(const char* file_name, int start_at, int end_at, int* input_size, int* output_size) {
    FILE *file = fopen(file_name, "r");
    // TODO maybe implement so that you can chose the first output and so on in the cvs file and maybe have the o1, i2 and so on determine the order instead of the placement in the cvs

    if (file == NULL) {
        char err_msg[40];
        sprintf(err_msg, "[LOG]: Error opening file '%s'\n", file_name);
        perror(err_msg);
        return NULL;
    }

    char line[100];
    char *token;

    int data_point_size = 0;

    (*input_size) = 0;
    (*output_size) = 0;

    while (fgets(line, sizeof(line), file) != NULL) {
        token = strtok(line, ",");
        while (token != NULL) {
            data_point_size++;
            token[strcspn(token, "\n")] = 0;
            if (strcmp(token, "i") == 0) {
                (*input_size)++;
            }
            if (strcmp(token, "o") == 0) {
                (*output_size)++;
            }
            token = strtok(NULL, ",");
        }
        break;
    }

    int num_elements = (int)((float)data_point_size * (float)(end_at - start_at + 1));
    float* dataset = (float*) malloc(sizeof(float) * num_elements);

    int line_ind = 1;
    int data_ind = 0;

    while (fgets(line, sizeof(line), file) != NULL) {
        token = strtok(line, ",");

        while (token != NULL) {
            if (line_ind >= start_at && line_ind <= end_at) {
                float token_val = (float)strtof(token, NULL);
//                printf("%f %d %d\n", token_val, line_ind, data_ind);
                dataset[data_ind++] = token_val;
            }

            token = strtok(NULL, ",");
        }
        line_ind++;
    }

    printf("[LOG] Read csv file %s from line %d to line %d. Input size = %d. Output size = %d. Elements total = %d\n",
            file_name, start_at, end_at, *input_size, *output_size, num_elements);

    fclose(file);

    return dataset;
}

// TODO serialization
void serialize_neural_network(Neural_Network* neural_network, const char* file_name) {
    FILE* fp = fopen(file_name, "wb");

    if (fp == NULL) {
        perror("[ERROR] Error opening file for serializing network.");
        return;
    }

    // NOTE i need the architecture, the weights and biases as well
    int total_size = neural_network->num_layers + neural_network->num_weights + neural_network->num_biases + 1;
    float* data = (float*) malloc(sizeof(float)*total_size);

    // NOTE specifically the number of layers is stored, from that all follows
    data[0] = (float)neural_network->num_layers;

    for (int i = 0; i < neural_network->num_layers; i++) {
        data[i + 1] = (float)neural_network->layer_sizes[i];
    }

    int j = 0;
    for (int i = neural_network->num_layers; i < neural_network->num_layers + neural_network->num_weights; i++) {
        data[i + 1] = neural_network->weights[j];
        j++;
    }

    j = 0;
    for (int i = neural_network->num_layers + neural_network->num_weights; i < neural_network->num_layers + neural_network->num_weights + neural_network->num_biases; i++) {
        data[i + 1] = neural_network->biases[j];
        j++;
    }

    fwrite(data, sizeof(float), total_size, fp);

    printf("\n[LOG] Serialized neural network in file %s.\n\n", file_name);

    fclose(fp);
}

Neural_Network* deserialize_neural_network(const char* file_name) {
    FILE* file = fopen(file_name, "rb");

    if (file == NULL) {
        perror("[ERROR] Error opening file for deserializing network.");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);

    int dataSize;
    dataSize = fileSize / sizeof(float);

    float* data = (float*)malloc(fileSize);

    fseek(file, 0, SEEK_SET);
    fread(data, sizeof(float), dataSize, file);

    fclose(file);

    int layer_sizes[(int)data[0]];
    for (int i = 0; i < (int)data[0]; i++) {
        layer_sizes[i] = (int)data[i+1];
    }

    Neural_Network *neural_network = initiate_neural_network(sizeof(layer_sizes) / sizeof(int), layer_sizes);

    int j = 0;
    for (int i = neural_network->num_layers; i < neural_network->num_layers + neural_network->num_weights; i++) {
        neural_network->weights[j] = data[i + 1];
        j++;
    }

    j = 0;
    for (int i = neural_network->num_layers + neural_network->num_weights; i < neural_network->num_layers + neural_network->num_weights + neural_network->num_biases; i++) {
        neural_network->biases[j] = data[i + 1];
        j++;
    }

    printf("\n[LOG] Deserialized neural network from file %s.\n\n", file_name);

    return neural_network;
}

// TODO minst dataset testing

int main()
{

    {
        int input_size, output_size;
        float *training_data = read_cvs_into_dataset("../../res/iris_dataset_training.csv", 1, 135, &input_size, &output_size);
        float *testing_data = read_cvs_into_dataset("../../res/iris_dataset_testing.csv", 1, 15, &input_size, &output_size);

        int layer_sizes[4] = {input_size, 5, 4, output_size};
        Neural_Network *neural_network = initiate_neural_network(sizeof(layer_sizes) / sizeof(int), layer_sizes);

        // ------------------------

        train_neural_network(neural_network, training_data, 135, 5.0f, 100000, 100000 - 1);

        float neural_network_cost = test_neural_network(neural_network, testing_data, 15);
        printf("[LOG] Cost on testing_data = %f\n", neural_network_cost);

        // NOTE example from iris_dataset_testing.csv, should return 1.0,0.0,0.0
        float inputs[4] = {5.0f, 3.4f, 1.5f, 0.2f};
        float outputs[3];
        evaluate_neural_network(neural_network, inputs, outputs, 1);
        printf("{%f %f %f %f} => {%f %f %f}\n", inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], outputs[1], outputs[2]);

        // ------------------------

        serialize_neural_network(neural_network, "iris.nn");

        free((void *) training_data);
        free((void *) testing_data);
        destroy_neural_network(neural_network);
    }

    {
        Neural_Network *neural_network = deserialize_neural_network("iris.nn");
        // NOTE example from iris_dataset_testing.csv, should return 1.0,0.0,0.0
        float inputs[4] = {5.0f, 3.4f, 1.5f, 0.2f};
        float outputs[3];
        evaluate_neural_network(neural_network, inputs, outputs, 1);
        printf("{%f %f %f %f} => {%f %f %f}\n", inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], outputs[1], outputs[2]);

        destroy_neural_network(neural_network);
    }

    return 0;
}
