#pragma once

#include <vector>
#include <string>

class NeuralLayer
{
public:
    size_t outputWidth, outputHeight, outputChannels;
    std::vector<float> outputs, outputGradients;

    NeuralLayer* previousLayer = nullptr;

    virtual void FeedForward() = 0;
    virtual void BackPropogate() = 0;
    virtual void Create(NeuralLayer* previousLayer) = 0;
    virtual size_t PrintStats() const = 0;

    void SetActivationFuction(std::string ActivationFunction);

    static void ReLu(NeuralLayer *NL);
    static void LeakyReLu(NeuralLayer* NL);

    static void ReLuDerivative(NeuralLayer* NL);
    static void LeakyReLuDerivative(NeuralLayer* NL);
    
    static void SoftMax(NeuralLayer* NL);
    static void SoftMaxDerivative(NeuralLayer* NL);

    NeuralLayer(size_t width, size_t height, size_t channels) :
        outputWidth(width), outputHeight(height), outputChannels(channels) {}
    NeuralLayer() :
        outputWidth(0), outputHeight(0), outputChannels(0) { }

    void (*Activation)(NeuralLayer*) = nullptr;
    void (*ActivationDerivative)(NeuralLayer*) = nullptr;

    float learningRate = 0.000015f;
};

class Input : public NeuralLayer
{
public:
    Input(size_t width, size_t height, size_t channels);

    void FeedForward() {};
    void BackPropogate() {};
    void Create(NeuralLayer* previousLayer) { outputGradients.assign(outputWidth * outputHeight * outputChannels, 0.0f); };
    size_t PrintStats() const;
};

class Convolution : public NeuralLayer
{
public:
    size_t kernelSize, padding, stride, kernelAmount;

    /*
    * The kernelweights consists of all the weights used by all the kernels in this layer.
    * It is stored in memory in rows, thus the first row that is stored, is the first row of the first channel of the first kernel,
    * After that comes the second row from the first kernel. Then the rows follow from the second channel and first kernel. 
    * After all the rows and channels of the first kernel, comes the first row of the second kernel
    */
    std::vector<float> kernelWeights, kernelGradients;

    /*
    * The biases for the kernels are stored here, thus at the firs index, the bias from the first kernel is stored.
    */
    std::vector<float> biasWeights, biasGradients;

    Convolution(size_t amount, size_t kernelSize, size_t padding = 0, size_t stride = 1, std::string ActivationFunction = "relu");

    void FeedForward();
    void BackPropogate();
    void Create(NeuralLayer* previousLayer);
    size_t PrintStats() const;

private:
    float CrossCorrelation(size_t beginX, size_t beginY, size_t kernel = 0) const;
    float WeightGradient(size_t beginX, size_t beginY, size_t kernel, size_t channel) const;
    inline float CalculateInputGradient(size_t c, size_t x, size_t y) const;
    inline float GetOutputGradientForBackPropogate(size_t x, size_t y) const;
};

class MaxPooling : public NeuralLayer
{
public:
    size_t poolingSize;

    MaxPooling(size_t poolSize = 2);

    void FeedForward();
    void BackPropogate();
    void Create(NeuralLayer* previousLayer);
    size_t PrintStats() const;

private:
    void Max(size_t i, size_t j, size_t k);

    /*
    * Is a vector with the same dimensions as the output, for every output it contains the 
    * index for the max input given, thus this index can be used in the previous layer's output.
    */
    std::vector<size_t> maxIndexes;
};

class FullyConnected : public NeuralLayer
{
public:
    /*
    * Contains all the weights for this connected layer, all the weights used by the first output
    * neuron are at the front of this vector. After that all the weights used by the second output neuron follow it.
    */
    std::vector<float> weights, weightGradients;
    std::vector<float> biasWeights, biasGradients;

    FullyConnected(size_t outputSize, std::string ActivationFunction = "relu");

    void FeedForward();
    void BackPropogate();
    void Create(NeuralLayer* previousLayer);
    size_t PrintStats() const;

private:
    size_t sizePreviousLayer = 0;
};