#include <stdlib.h>
#include <stdbool.h>

typedef struct Neuron{
    void** inputs;
    double *weights;
    double *deltaWeights;
    double output;
    double error;
    double (*activationFunction)(double);
    double (*activationFunctionDerivative)(double);
} *Neuron_t;

Neuron_t createNeuron(int numInputs, double (*activationFunction)(double), double (*activationFunctionDerivative)(double));

void destroyNeuron(Neuron_t neuron);

typedef struct Layer{
    bool simple;
    Neuron_t *neurons;
    int numNeurons;
} *Layer_t;

Layer_t createSimpleLayer(int numNeurons, int numInputs, double (*activationFunction)(double), double (*activationFunctionDerivative)(double));

void destroyLayer(Layer_t layer);

typedef struct Network{
    Layer_t *layers;
    int numLayers;
} *Network_t;

Network_t createNetwork(int numLayers, int *numNeuronsPerLayer, double (*activationFunction)(double), double (*activationFunctionDerivative)(double));

void destroyNetwork(Network_t network);


