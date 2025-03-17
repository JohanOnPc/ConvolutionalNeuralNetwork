#pragma once

#include <vector>

class NeuralLayer
{
public:
    size_t outputWidth, outputHeight, outputChannels;
    std::vector<float> outputs;

    NeuralLayer* previousLayer = nullptr;

    virtual void FeedForward() = 0;
    virtual void BackPropogate() = 0;
    virtual void Create(NeuralLayer* previousLayer) = 0;

    NeuralLayer(size_t width, size_t height, size_t channels) :
        outputWidth(width), outputHeight(height), outputChannels(channels) {}
    NeuralLayer() :
        outputWidth(0), outputHeight(0), outputChannels(0) { }
};

class Input : NeuralLayer
{
public:
    Input(size_t width, size_t height, size_t channels);

    void FeedForward() {};
    void BackPropogate() {};
    void Create(NeuralLayer *previousLayer) {};
};

class Convolution : NeuralLayer
{
public:
    size_t kernelSize, padding, stride, kernelAmount;

    /*
    * The kernelweights consists of all the weights used by all the kernels in this layer.
    * It is stored in memory in rows, thus the first row that is stored, is the first row of the first kernel,
    * After that comes the second row from the first kernel. After all the rows of the first kernel, 
    * comes the first row of the second kernel
    */
    std::vector<float> kernelWeights;

    /*
    * The biases for the kernels are stored here, thus at the firs index, the bias from the first kernel is stored.
    */
    std::vector<float> biasWeights;

    Convolution(size_t amount, size_t kernelSize, size_t padding = 0, size_t stride = 1);

    void FeedForward();
    void BackPropogate();
    void Create(NeuralLayer* previousLayer);

private:
    float CrossCorrelation(size_t beginX, size_t beginY, size_t kernel = 0) const;
};

class MaxPooling : NeuralLayer
{
public:
    size_t poolingSize;

    MaxPooling(size_t poolSize = 2);

    void FeedForward();
    void BackPropogate();
    void Create(NeuralLayer* previousLayer);

private:
    void Max(size_t i, size_t j, size_t k);

    /*
    * Is a vector with the same dimensions as the output, for every output it contains the 
    * index for the max input given, thus this index can be used in the previous layer's output.
    */
    std::vector<size_t> maxIndexes;
};

class FullyConnected : NeuralLayer
{
public:
    std::vector<float> weights;
    std::vector<float> biasWeights;

    FullyConnected(size_t outputSize);

    void FeedForward();
    void BackPropogate();
    void Create(NeuralLayer* previousLayer);

private:
    size_t sizePreviousLayer = 0;
};