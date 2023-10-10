#include "makina.h"
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

        train_neural_network(neural_network, training_data, 135, 5.0f, 100000, 100000);

        float neural_network_cost = test_neural_network(neural_network, testing_data, 15);
        printf("[LOG] Cost on testing_data = %f\n", neural_network_cost);

        // NOTE example from iris_dataset_testing.csv, should return 1.0,0.0,0.0
        float inputs[4] = {5.0f, 3.4f, 1.5f, 0.2f};
        float outputs[3];
        evaluate_neural_network(neural_network, inputs, outputs, 1);
        printf("{%f %f %f %f} => {%f %f %f}\n", inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], outputs[1], outputs[2]);

        // ------------------------

        serialize_neural_network(neural_network, "../../res/iris.nn");

        free((void *) training_data);
        free((void *) testing_data);
        destroy_neural_network(neural_network);
    }

    {
        Neural_Network *neural_network = deserialize_neural_network("../../res/iris.nn");
        // NOTE example from iris_dataset_testing.csv, should return 1.0,0.0,0.0
        float inputs[4] = {5.0f, 3.4f, 1.5f, 0.2f};
        float outputs[3];
        evaluate_neural_network(neural_network, inputs, outputs, 1);
        printf("{%f %f %f %f} => {%f %f %f}\n", inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], outputs[1], outputs[2]);

        destroy_neural_network(neural_network);
    }

    return 0;
}
