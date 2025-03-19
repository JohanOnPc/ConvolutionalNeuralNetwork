#include "NeuralNetwork.h"

int main()
{
    NeuralNetwork *model = new NeuralNetwork();
    model->AddLayer(new Input(28, 28, 1));
    model->AddLayer(new Convolution(5, 5, 2));
    model->AddLayer(new MaxPooling(2));
    model->AddLayer(new Convolution(5, 5, 2));
    model->AddLayer(new MaxPooling(2));
    model->AddLayer(new FullyConnected(100));
    model->AddLayer(new FullyConnected(30));
    model->AddLayer(new FullyConnected(10, "softmax"));

    model->Create();

    model->PrintSummary();

    return 0;
}