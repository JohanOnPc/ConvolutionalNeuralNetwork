#include "NeuralLayer.h"

Input::Input(size_t width, size_t height, size_t channels) : 
    NeuralLayer(width, height, channels)
{
    outputs.reserve(width * height * channels);
}

Convolution::Convolution(size_t amount, size_t kernelSize, size_t padding, size_t stride) :
    kernelSize(kernelSize), padding(padding), stride(stride), kernelAmount(amount)
{
    biasWeights.reserve(amount);
}

void Convolution::FeedForward()
{
    for (size_t k = 0; k < kernelAmount; k++)
    {
        for (size_t j = 0; j < previousLayer->outputHeight; j++)
        {
            for (size_t i = 0; i < previousLayer->outputWidth; i++)
            {

            }
        }
    }
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
    kernelWeights.reserve(kernelAmount * previousLayer->outputChannels * kernelSize * kernelSize);
}

float Convolution::CrossCorrelation(size_t beginX, size_t beginY, size_t kernel = 0) const
{
    float kernelBase = kernel * kernelSize * kernelSize;
    float sum = 0;
    for (size_t k = 0; k <= previousLayer->outputChannels; k++) {
        for (size_t y = 0; y < kernelSize; y++) {
            float Y = y + beginY;
            for (size_t x = 0; x < + kernelSize; x++) {
                float X = x + beginX;

                float input = previousLayer->outputs[kernel * previousLayer->outputWidth * outputHeight + Y * outputWidth + X];
                float weight = kernelWeights[kernelBase + y * kernelSize + x];
                sum += input * weight;
            }
        }
    }

    return sum;
}
