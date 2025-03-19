#include "NeuralNetwork.h"

#include <iostream>

void NeuralNetwork::AddLayer(NeuralLayer* layer)
{
	Layers.push_back(layer);
}

void NeuralNetwork::Create()
{
	NeuralLayer* previousLayer = nullptr;
	for (auto& layer : Layers)
	{
		layer->Create(previousLayer);
		previousLayer = layer;
	}
}

void NeuralNetwork::PrintSummary() const
{
	size_t totalParams = 0;

	for (auto& layer : Layers)
	{
		totalParams += layer->PrintStats();
	}

	std::cout << "Total Trainable params: " << totalParams << '\n';
}
