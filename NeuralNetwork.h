#pragma once

#include <vector>
#include <memory>

#include "NeuralLayer.h"
#include "common.h"

class NeuralNetwork
{
private:
    std::vector<NeuralLayer*> Layers;
    
public:
    NeuralNetwork() {}
    NeuralNetwork(std::vector<NeuralLayer*> layer) {}

    void AddLayer(NeuralLayer* layer);

    std::vector<float> Predict(const std::vector<float> &Input);

    void Create();
    void PrintSummary() const;
    //void Fit(size_t epochs, const struct dataSet& dataSet);
    void Fit(size_t epochs, const std::vector<std::vector<float>>& trainInput, const std::vector<size_t>& trainLabels, const std::vector<std::vector<float>>& validationInput, const std::vector<size_t>& validationLabels);

private:
    void BackPropogate(const std::vector<float>& expected);
    inline void FeedForward();
};
 
