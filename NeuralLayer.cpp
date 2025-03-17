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
    for (size_t k = 0; k < kernelAmount; k++) {
        for (size_t j = 0; j < outputHeight; j++) {
            for (size_t i = 0; i < outputWidth; i++) {
                float Z = CrossCorrelation(i, j, k) + biasWeights[k];

                //activation function

                outputs[k * kernelSize * kernelSize + j * kernelSize + i] = Z;
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


/*
* Cross Corelates the kernel given by the index, with the previous layer's output based upon
* The given x and y coordinates.
*/
float Convolution::CrossCorrelation(size_t beginX, size_t beginY, size_t kernel) const
{
    size_t kernelBase = kernel * kernelSize * kernelSize;
    float sum = 0;
    for (size_t k = 0; k <= previousLayer->outputChannels; k++) {
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
    maxIndexes.reserve(outputWidth * outputHeight * outputChannels);
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
            if (max < previousLayer->outputs[inputK + y + x]) {
                index = inputK + y + x;
                max = previousLayer->outputs[index];
            }
        }
    }

    size_t outputIndex = k * outputWidth * outputHeight + j * outputWidth + i;
    outputs[outputIndex] = max;
    maxIndexes[outputIndex] = index;
}

FullyConnected::FullyConnected(size_t outputSize)
{
    outputChannels = 1;
    outputWidth = 1;
    outputHeight = outputSize;

    biasWeights.reserve(outputSize);
}

void FullyConnected::FeedForward()
{
    for (size_t k = 0; k < outputHeight; k++) {
        float Z = 0;

        for (size_t j = 0; j < sizePreviousLayer; j++) {
            Z += previousLayer->outputs[j] * weights[k * outputHeight + j];
        }

        Z += biasWeights[k];

        //Activation Function

        outputs[k] = Z;
    }
}

void FullyConnected::BackPropogate()
{
}

void FullyConnected::Create(NeuralLayer* previousLayer)
{
    this->previousLayer = previousLayer;
    sizePreviousLayer = previousLayer->outputWidth * previousLayer->outputHeight * previousLayer->outputChannels;

    weights.reserve(outputHeight * sizePreviousLayer);
}
