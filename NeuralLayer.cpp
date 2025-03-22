#include "NeuralLayer.h"
#include "common.h"

#include <iostream>
#include <algorithm>
#include <ranges>
#include <numeric>
#include <format>

void NeuralLayer::SetActivationFuction(std::string ActivationFunction)
{
    if (ActivationFunction == "relu")
        Activation = ReLu;
    else if (ActivationFunction == "softmax")
        Activation = SoftMax;
    else
    {
        std::cerr << "Given Activation function: '" << ActivationFunction << "' does not exists!\n Exiting!";
        exit(1);
    }
}

void NeuralLayer::ReLu(NeuralLayer* NL)
{
    std::transform(NL->outputs.begin(), NL->outputs.end(), NL->outputs.begin(), [](float Z) {return std::max(0.f, Z); });
}

void NeuralLayer::ReLuDerivative(NeuralLayer* NL)
{
    for (size_t i = 0; i < NL->outputs.size(); i++) {
        NL->outputGradients[i] *= (NL->outputs[i] > 0.f);
    }
}

void NeuralLayer::SoftMax(NeuralLayer* NL)
{
    auto max = std::max_element(NL->outputs.cbegin(), NL->outputs.cend());
    std::transform(NL->outputs.begin(), NL->outputs.end(), NL->outputs.begin(), [max](float Z) {return Z - *max; });

    float sum = std::accumulate(NL->outputs.cbegin(), NL->outputs.cend(), 0.f, [](float acc, float Z) {return acc + std::expf(Z); });

    std::transform(NL->outputs.begin(), NL->outputs.end(), NL->outputs.begin(), [&sum](float Z) {return std::expf(Z) / sum; });
}

Input::Input(size_t width, size_t height, size_t channels) : 
    NeuralLayer(width, height, channels)
{
    outputs.reserve(width * height * channels);
}

size_t Input::PrintStats() const
{
    std::cout << std::format("Input [{}, {}, {}]\n", outputWidth, outputHeight, outputChannels);

    return 0;
}

Convolution::Convolution(size_t amount, size_t kernelSize, size_t padding, size_t stride, std::string ActivationFunction) :
    kernelSize(kernelSize), padding(padding), stride(stride), kernelAmount(amount)
{
    biasWeights.reserve(amount);

    SetActivationFuction(ActivationFunction);
}

void Convolution::FeedForward()
{
    for (size_t k = 0; k < kernelAmount; k++) {
        for (size_t j = 0; j < outputHeight; j++) {
            for (size_t i = 0; i < outputWidth; i++) {
                float Z = CrossCorrelation(i, j, k) + biasWeights[k];

                outputs[k * kernelSize * kernelSize + j * kernelSize + i] = Z;
            }
        }
    }

    Activation(this);
}

void Convolution::BackPropogate()
{
}

void Convolution::Create(NeuralLayer* previousLayer)
{
    this->previousLayer = previousLayer;

    outputWidth = (previousLayer->outputWidth + 2 * padding - kernelSize) / stride + 1;
    outputHeight = (previousLayer->outputHeight + 2 * padding - kernelSize) / stride + 1;
    outputChannels = kernelAmount;

    outputs.reserve(outputWidth * outputHeight * outputChannels);
    outputs.assign(outputWidth * outputHeight * outputChannels, 0.0);
    kernelWeights.reserve(kernelAmount * previousLayer->outputChannels * kernelSize * kernelSize);

    outputGradients.assign(outputWidth * outputHeight * outputChannels, 0.f);

    InitWeights(biasWeights, kernelAmount,kernelSize * kernelSize);
    InitWeights(kernelWeights, kernelSize * kernelSize * kernelAmount, kernelSize * kernelSize);
}

size_t Convolution::PrintStats() const
{
    size_t params = kernelWeights.size() + biasWeights.size();

    std::cout << std::format("Convolution [{}, {}, {}] {}\n", outputWidth, outputHeight, outputChannels, params);

    return params;
}


/*
* Cross Corelates the kernel given by the index, with the previous layer's output based upon
* The given x and y coordinates.
*/
float Convolution::CrossCorrelation(size_t beginX, size_t beginY, size_t kernel) const
{
    size_t kernelBase = kernel * kernelSize * kernelSize;
    float sum = 0;
    for (size_t k = 0; k < previousLayer->outputChannels; k++) {
        for (size_t y = 0; y < kernelSize; y++) {
            size_t Y = y + beginY;
            for (size_t x = 0; x < + kernelSize; x++) {
                size_t X = x + beginX;

                float input = previousLayer->outputs[k * previousLayer->outputWidth * previousLayer->outputHeight + Y * previousLayer->outputWidth + X];
                float weight = kernelWeights[kernelBase + y * kernelSize + x];
                sum += input * weight;
            }
        }
    }

    return sum;
}

MaxPooling::MaxPooling(size_t poolSize) :
    poolingSize(poolSize)
{
}

void MaxPooling::FeedForward()
{
    for (size_t k = 0; k < outputChannels; k++) {
        for (size_t j = 0; j < outputHeight; j++) {
            for (size_t i = 0; i < outputWidth; i++) {
                Max(i, j, k);
            }
        }
    }
}

void MaxPooling::BackPropogate()
{
}

void MaxPooling::Create(NeuralLayer* previousLayer)
{
    this->previousLayer = previousLayer;

    outputWidth = previousLayer->outputWidth / poolingSize;
    outputHeight = previousLayer->outputHeight / poolingSize;
    outputChannels = previousLayer->outputChannels;

    outputs.reserve(outputWidth * outputHeight * outputChannels);
    outputs.assign(outputWidth * outputHeight * outputChannels, 0.0f);
    maxIndexes.reserve(outputWidth * outputHeight * outputChannels);
    maxIndexes.assign(outputWidth * outputHeight * outputChannels, 0);

    outputGradients.assign(outputWidth * outputHeight * outputChannels, 0.f);
}

size_t MaxPooling::PrintStats() const
{
    std::cout << std::format("MaxPooling [{}, {}, {}] {}\n", outputWidth, outputHeight, outputChannels, 0);

    return 0;
}

/*
* Calcules the max value based upon the previous layer's output. 
* the i, j, and k values are given for this layer itself.
*/
void MaxPooling::Max(size_t i, size_t j, size_t k)
{
    size_t inputK = k * previousLayer->outputWidth * previousLayer->outputHeight;
    size_t inputJ = inputK + j * poolingSize * previousLayer->outputWidth;
    size_t inputI = inputJ + i * poolingSize;

    float max = std::numeric_limits<float>::lowest(); //set to lowest possible value for floats.
    size_t index = 0;

    for (size_t Y = 0; Y < poolingSize; Y++) {
        size_t y = inputJ + Y * previousLayer->outputWidth;

        for (size_t x = 0; x < poolingSize; x++) {
            if (max < previousLayer->outputs[y + x]) {
                index = y + x;
                max = previousLayer->outputs[index];
            }
        }
    }

    size_t outputIndex = k * outputWidth * outputHeight + j * outputWidth + i;
    outputs[outputIndex] = max;
    maxIndexes[outputIndex] = index;
}

FullyConnected::FullyConnected(size_t outputSize, std::string ActivationFunction)
{
    outputChannels = 1;
    outputWidth = 1;
    outputHeight = outputSize;

    biasWeights.reserve(outputSize);

    SetActivationFuction(ActivationFunction);
}

void FullyConnected::FeedForward()
{
    for (size_t k = 0; k < outputHeight; k++) {
        float Z = 0;

        for (size_t j = 0; j < sizePreviousLayer; j++) {
            Z += previousLayer->outputs[j] * weights[k * sizePreviousLayer + j];
        }

        Z += biasWeights[k];

        outputs[k] = Z;
    }

    Activation(this);
}

void FullyConnected::BackPropogate()
{
    //Gradient with respect to the output after activation
    //Calculate the gradient with respect to the output based on the derivative of the used activation fucntion. 
    ReLuDerivative(this);

    //Gradient with respect to the weights

    for (size_t k = 0; k < outputHeight; k++) {
        for (size_t j = 0; j < sizePreviousLayer; j++) {
            weightGradients[k * sizePreviousLayer + j] = previousLayer->outputs[j] * outputGradients[k];
        }
    }

    //Gradient with respect to the bias

    for (size_t i = 0; i < outputHeight; i++)
        biasGradients[i] = outputGradients[i];

    //Gradient with respect to the input

    for (size_t j = 0; j < sizePreviousLayer; j++) {
        float gradient = 0.f;

        for (size_t k = 0; k < outputHeight; k++) {
            gradient += outputGradients[k] * weights[k * sizePreviousLayer + j];
        }

        previousLayer->outputGradients[j] = gradient;
    }

    for (size_t k = 0; k < outputHeight * sizePreviousLayer; k++) {
        weights[k] -= learningRate * weightGradients[k];
    }

    for (size_t b = 0; b < outputHeight; b++) {
        biasWeights[b] -= learningRate * biasGradients[b];
    }
}

void FullyConnected::Create(NeuralLayer* previousLayer)
{
    this->previousLayer = previousLayer;
    sizePreviousLayer = previousLayer->outputWidth * previousLayer->outputHeight * previousLayer->outputChannels;

    weights.reserve(outputHeight * sizePreviousLayer);
    outputs.assign(outputHeight, 0.f);

    outputGradients.assign(outputHeight, 0.f);
    weightGradients.assign(outputHeight * sizePreviousLayer, 0.f);
    biasGradients.assign(outputHeight, 0.f);

    InitWeights(weights, outputHeight * sizePreviousLayer, sizePreviousLayer);
    InitWeights(biasWeights, outputHeight,sizePreviousLayer);
}

size_t FullyConnected::PrintStats() const
{
    size_t params = weights.size() + biasWeights.size();

    std::cout << std::format("FullyConnected [{}] {}\n", outputHeight, params);

    return params;
}
