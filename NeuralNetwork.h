#pragma once

#include <vector>
#include <memory>

#include "NeuralLayer.h"

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
    void Fit(const std::vector<std::vector<float>>& Input, const std::vector<size_t>& labels);

private:
    void BackPropogate(const std::vector<float>& expected);
    inline void FeedForward();
};
 
