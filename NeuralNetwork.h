#pragma once

#include <vector>
#include <memory>

#include "NeuralLayer.h"
#include "common.h"

class NeuralNetwork
{
private:
    std::vector<NeuralLayer*> Layers;
    float learningRate{};
    float decayRate{};
    
public:
    NeuralNetwork() {}
    NeuralNetwork(std::vector<NeuralLayer*> layer) {}

    void AddLayer(NeuralLayer* layer);

    std::vector<float> Predict(const std::vector<float> &Input);

    void Create(float learningRate = 0.000015f, float decayRate = 0.f);
    void PrintSummary() const;
    void Fit(size_t epochs, const struct DataSet& dataSet);
    void Fit(size_t epochs, const std::vector<std::vector<float>>& trainInput, const std::vector<size_t>& trainLabels, const std::vector<std::vector<float>>& validationInput, const std::vector<size_t>& validationLabels);

    void SetLearningRate(float learningRate) const;

    void SaveModel(const std::string& fileName) const;
    void LoadModel(const std::string& fileName);

private:
    void BackPropogate(const std::vector<float>& expected);
    inline void FeedForward();
};
 
