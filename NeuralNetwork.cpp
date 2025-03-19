#include "NeuralNetwork.h"

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
