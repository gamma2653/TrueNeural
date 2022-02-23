#ifndef NEURAL_NET_C
#define NEURAL_NET_C
#include "neural_net.h"



Neuron_t createNeuron(int numInputs, double (*activationFunction)(double), double (*activationFunctionDerivative)(double)) {
    Neuron_t neuron = malloc(sizeof(struct Neuron));
    neuron->inputs = NULL;
    neuron->weights = malloc(sizeof(double) * numInputs);
    neuron->deltaWeights = malloc(sizeof(double) * numInputs);
    neuron->activationFunction = activationFunction;
    neuron->activationFunctionDerivative = activationFunctionDerivative;
    neuron->error = 0;
    neuron->output = 0;
    return neuron;
}

void destroyNeuron(Neuron_t neuron) {
    free(neuron->weights);
    free(neuron->deltaWeights);
    free(neuron);
}

Layer_t createSimpleLayer(int numNeurons, int numInputs, double (*activationFunction)(double), double (*activationFunctionDerivative)(double)) {
    Layer_t layer = malloc(sizeof(struct Layer));
    layer->neurons = malloc(sizeof(Neuron_t) * numNeurons);
    layer->numNeurons = numNeurons;
    layer->simple = true;
    for (int i = 0; i < numNeurons; i++) {
        layer->neurons[i] = createNeuron(numInputs, activationFunction, activationFunctionDerivative);
    }
    return layer;
}

Layer_t createComplexLayer(int numNeurons, int numInputs, int numOutputs, double (*activationFunction)(double), double (*activationFunctionDerivative)(double)) {
    Layer_t layer = malloc(sizeof(struct Layer));
    layer->neurons = malloc(sizeof(Neuron_t) * numNeurons);
    layer->numNeurons = numNeurons;
    layer->simple = false;
    
    return layer;
}

void destroyLayer(Layer_t layer) {
    for (int i = 0; i < layer->numNeurons; i++) {
        destroyNeuron(layer->neurons[i]);
    }
    free(layer->neurons);
    free(layer);
}

Network_t createNetwork(int numLayers, int *numNeuronsPerLayer, double (*activationFunction)(double), double (*activationFunctionDerivative)(double)) {
    Network_t network = malloc(sizeof(struct Network));
    network->layers = malloc(sizeof(Layer_t) * numLayers);
    network->numLayers = numLayers;
    for (int i = 0; i < numLayers; i++) {
        network->layers[i] = createLayer(numNeuronsPerLayer[i], i == 0 ? 0 : numNeuronsPerLayer[i - 1], activationFunction, activationFunctionDerivative);
    }
    return network;
}

void destroyNetwork(Network_t network) {
    for (int i = 0; i < network->numLayers; i++) {
        destroyLayer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}

#endif

