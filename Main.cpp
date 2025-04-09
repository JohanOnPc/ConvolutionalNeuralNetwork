#include "NeuralNetwork.h"
#include "common.h"
#include "MNISTreader.h"

int main()
{
    /*
    NeuralNetwork* model = new NeuralNetwork();
    model->AddLayer(new Input(28, 28, 1));
    model->AddLayer(new Convolution(8, 5, 0, 1, "relu"));
    model->AddLayer(new MaxPooling(2));
    model->AddLayer(new Convolution(16, 3, 0, 1, "relu"));
    model->AddLayer(new MaxPooling(2));
    model->AddLayer(new FullyConnected(100, "relu"));
    model->AddLayer(new FullyConnected(30, "relu"));
    model->AddLayer(new FullyConnected(10, "softmax"));

    model->Create(3E-4f, 0.1f);

    model->PrintSummary();
    */
    
    NeuralNetwork* model = new NeuralNetwork();
    model->AddLayer(new Input(28, 28, 1));
    model->AddLayer(new FullyConnected(128, "relu"));
    model->AddLayer(new FullyConnected(64, "relu"));
    model->AddLayer(new FullyConnected(64, "relu"));
    model->AddLayer(new FullyConnected(10, "softmax"));

    model->Create(6E-5f, 0.1f);
    model->PrintSummary();

    DataSet _dataSet = ReadMNISTDataSet("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte", "dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");

    model->Fit(20, _dataSet);

    model->SaveModel("better.model");

    NeuralNetwork* model2 = new NeuralNetwork();

    model2->LoadModel("better.model");

    model2->PrintSummary();

    model2->Fit(1, _dataSet);

    return 0;
}