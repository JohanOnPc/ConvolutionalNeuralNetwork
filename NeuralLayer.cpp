#include "NeuralLayer.h"
#include "common.h"

#include <iostream>
#include <algorithm>
#include <ranges>
#include <numeric>
#include <format>

void NeuralLayer::SetActivationFuction(std::string ActivationFunction)
{
    if (ActivationFunction == "relu") {
        Activation = ReLu;
        ActivationDerivative = ReLuDerivative;
    }
    else if (ActivationFunction == "softmax") {
        Activation = SoftMax;
        ActivationDerivative = SoftMaxDerivative;
    }
    else if (ActivationFunction == "leakyrelu") {
        Activation = LeakyReLu;
        ActivationDerivative = LeakyReLuDerivative;
    }
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

void NeuralLayer::LeakyReLu(NeuralLayer* NL)
{
    std::transform(NL->outputs.begin(), NL->outputs.end(), NL->outputs.begin(), [](float Z) {return std::max(0.1f * Z, Z); });
}

void NeuralLayer::ReLuDerivative(NeuralLayer* NL)
{
    for (size_t i = 0; i < NL->outputs.size(); i++) {
        NL->outputGradients[i] *= (NL->outputs[i] > 0.f);
    }
}

void NeuralLayer::LeakyReLuDerivative(NeuralLayer* NL)
{
    for (size_t i = 0; i < NL->outputs.size(); i++) {
        NL->outputGradients[i] *= ((NL->outputs[i] > 0.f) + (NL->outputs[i] <= 0.f) * 0.1f);
    }
}

void NeuralLayer::SoftMax(NeuralLayer* NL)
{
    auto max = std::max_element(NL->outputs.cbegin(), NL->outputs.cend());
    std::transform(NL->outputs.begin(), NL->outputs.end(), NL->outputs.begin(), [max](float Z) {return std::clamp(Z - *max, -80.f, 50.f); });

    float sum = std::accumulate(NL->outputs.cbegin(), NL->outputs.cend(), 0.f, [](float acc, float Z) {return acc + std::expf(Z); });

    sum = std::max(sum, 1E-12f);

    std::transform(NL->outputs.begin(), NL->outputs.end(), NL->outputs.begin(), [&sum](float Z) {return std::expf(Z) / sum; });
}

void NeuralLayer::SoftMaxDerivative(NeuralLayer* NL)
{
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
    ActivationDerivative(this);

    //Gradient with respect to the weights

    for (size_t kernel = 0; kernel < kernelAmount; kernel++) {
        for (size_t channel = 0; channel < previousLayer->outputChannels; channel++) {
            for (size_t y = 0; y < kernelSize; y++) {
                for (size_t x = 0; x < kernelSize; x++) {
                    float gradient = WeightGradient(x, y, kernel, channel);
                    kernelGradients[kernel * previousLayer->outputChannels * kernelSize * kernelSize + channel * kernelSize * kernelSize + y * kernelSize + x] = gradient;
                }
            }
        }
    }

    //Gradient with respect to the bias
    

    for (size_t k = 0; k < kernelAmount; k++) {
        float biasGradient = 0.f;

        for (size_t i = 0; i < outputWidth * outputHeight; i++) {
            biasGradient += outputGradients[k * outputHeight * outputWidth + i];
        }

        biasGradients[k] = biasGradient;
    }

    //Gradient with respect to the input
    for (size_t c = 0; c < previousLayer->outputChannels; c++) {
        for (size_t y = 0; y < previousLayer->outputHeight; y++) {
            for (size_t x = 0; x < previousLayer->outputWidth; x++) {
                float inputGradient = CalculateInputGradient(c, x, y);

                previousLayer->outputGradients[c * previousLayer->outputWidth * previousLayer->outputHeight + y * previousLayer->outputWidth + x] = inputGradient;
            }
        }
    }

    //update all the weights based upon the gradients

    for (size_t i = 0; i < kernelWeights.size(); i++) {
        kernelWeights[i] -= learningRate * kernelGradients[i];
    }

    for (size_t k = 0; k < kernelAmount; k++) {
        biasWeights[k] -= learningRate * biasGradients[k];
    }
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
    kernelGradients.assign(kernelAmount * previousLayer->outputChannels * kernelSize * kernelSize, 0.f);
    biasGradients.assign(kernelAmount, 0.f);

    InitWeights(biasWeights, kernelAmount, kernelSize * kernelSize);
    InitWeights(kernelWeights, kernelSize * kernelSize * kernelAmount * previousLayer->outputChannels, kernelSize * kernelSize);
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
    /*size_t kernelBase = kernel * kernelSize * kernelSize;
    float sum = 0.f;
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
    }*/

    float sum = 0.f;
    size_t kernelBase = kernel * kernelSize * kernelSize * previousLayer->outputChannels;

    for (size_t k = 0; k < previousLayer->outputChannels; k++) {
        size_t channelBase = k * previousLayer->outputWidth * previousLayer->outputHeight;

        for (size_t y = 0; y < kernelSize; y++) {
            size_t inputY = beginY + y;

            for (size_t x = 0; x < kernelSize; x++) {
                size_t inputX = beginX + x;

                float input = previousLayer->outputs[channelBase + inputY * previousLayer->outputWidth + inputX];
                float weight = kernelWeights[kernelBase + k * kernelSize * kernelSize + y * kernelSize + x];
                sum += input * weight;
            }
        }
    }

    return sum;
}

/*
* Calculates the weight gradient for the given weight using the local x, y and kernel coordinates. The channel coordinate gives the output channel of the previous layer.
*/
float Convolution::WeightGradient(size_t beginX, size_t beginY, size_t kernel, size_t channel) const
{
    float sum = 0.f;

    for (size_t y = 0; y < outputHeight; y++) {
        for (size_t x = 0; x < outputWidth; x++) {
            float input = previousLayer->outputs[channel * previousLayer->outputWidth * previousLayer->outputHeight + (beginY + y) * previousLayer->outputWidth + (beginX + x)];
            float outputGradient = outputGradients[kernel * outputWidth * outputHeight + y * outputWidth + x];

            sum += input * outputGradient;
        }
    }

    return sum;
}

inline float Convolution::CalculateInputGradient(size_t c, size_t x, size_t y) const
{
    size_t pad = kernelSize - 1 - padding;  //logical padding around the output gradients in all directions
    size_t deltaPad = (previousLayer->outputWidth - outputWidth) / 2; //Difference in padding between the input and output
    float gradient = 0.f;

    auto kernelChunks = kernelWeights | std::views::chunk(kernelSize * kernelSize);

    for (size_t k = 0; k < kernelAmount; k++) {
        auto rotatedKernelWeights = std::views::reverse(kernelChunks[k * previousLayer->outputChannels + c]);

        for (size_t localY = std::max(pad - y, 0ull); localY < std::min(kernelSize, previousLayer->outputHeight - y); localY++) {
            for (size_t localX = std::max(pad - x, 0ull); localX < std::min(kernelSize, previousLayer->outputWidth - x); localX++) {
                float kernelWeight = rotatedKernelWeights[y * kernelSize + x];
                size_t outputX = localX + x - pad;
                size_t outputY = localY + y - pad;
                float outputGradient = outputGradients[k * outputWidth * outputHeight + outputY * outputWidth + outputX];
                gradient += kernelWeight * outputGradient;
            }
        }
    }

    return gradient;
}

inline float Convolution::GetOutputGradientForBackPropogate(size_t x, size_t y) const
{


    return 0.0f;
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
    //Reset all the gradients for the input
    std::fill(previousLayer->outputGradients.begin(), previousLayer->outputGradients.end(), 0.f);

    //Set the gradient for the inputs only for those with the max value that was given to this output;
    for (size_t i = 0; i < outputWidth * outputHeight * outputChannels; i++) {
        previousLayer->outputGradients[maxIndexes[i]] = outputGradients[i];
    }
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

    float max = std::numeric_limits<float>::lowest(); //set to lowest possible value for floats.
    size_t index = 0;

    for (size_t y = 0; y < poolingSize; y++) {
        size_t inputY = j * poolingSize + y;

        for (size_t x = 0; x < poolingSize; x++) {
            size_t inputX = i * poolingSize + x;

            size_t inputIndex = inputK + inputY * previousLayer->outputWidth + inputX;

            if (max < previousLayer->outputs[inputIndex]) {
                index = inputIndex;
                max = previousLayer->outputs[inputIndex];
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
    ActivationDerivative(this);

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


    //update all the weights based upon the gradients
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
