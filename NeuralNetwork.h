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

    void Create();
};
 
