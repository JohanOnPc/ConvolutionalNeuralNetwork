#include "NeuralNetwork.h"
#include "common.h"
#include "MNISTreader.h"

int main()
{
    NeuralNetwork *model = new NeuralNetwork();
    model->AddLayer(new Input(28, 28, 1));
    model->AddLayer(new Convolution(8, 5));
    model->AddLayer(new MaxPooling(2));
    model->AddLayer(new Convolution(16, 5));
    model->AddLayer(new MaxPooling(2));
    model->AddLayer(new FullyConnected(100));
    model->AddLayer(new FullyConnected(30));
    model->AddLayer(new FullyConnected(10, "softmax"));

    model->Create();

    model->PrintSummary();

    auto output = model->Predict(std::vector<float>(28*28, 0.5f));
    PrintVector(output);

    dataSet _dataSet = ReadMNISTDataSet("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte", "dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");

    model->Fit(10, _dataSet);

    return 0;
}