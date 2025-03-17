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
};

class Convolution : NeuralLayer
{
public:
    size_t kernelSize, padding, stride, kernelAmount;
    std::vector<float> kernelWeights;
    std::vector<float> biasWeights;

    Convolution(size_t amount, size_t kernelSize, size_t padding = 0, size_t stride = 1);

    void FeedForward();
    void BackPropogate();
    void Create(NeuralLayer* previousLayer);

private:
    float CrossCorrelation(size_t beginX, size_t beginY, size_t kernel = 0) const;
};