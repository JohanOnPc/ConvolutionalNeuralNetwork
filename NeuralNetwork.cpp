#include "NeuralNetwork.h"
#include "common.h"

#include <iostream>
#include <ranges>

void NeuralNetwork::AddLayer(NeuralLayer* layer)
{
	Layers.push_back(layer);
}

std::vector<float> NeuralNetwork::Predict(const std::vector<float>& Input)
{
	if (Input.size() != (Layers[0]->outputChannels * Layers[0]->outputHeight * Layers[0]->outputWidth)) {
		std::cout << "Error Predict(), Given input is not the same size as the expected output!\n";
		exit(1);
	}

	Layers[0]->outputs = Input;

	FeedForward();

	return Layers.back()->outputs;
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

void NeuralNetwork::Fit(const std::vector<std::vector<float>>& input, const std::vector<size_t>& labels)
{
	//set input to data
	//feed forward through all the layers
	//Calculate the error
	//Propogate the gradients throughout the network

	if (input.size() != labels.size()) {
		std::cout << "Error Fit(), Amount of Inputs is not the same as the amount of labels\n";
		exit(1);
	}

	for (size_t n = 0; n < input.size(); n++) {
		Layers.front()->outputs = input[n];
		FeedForward();

		auto expectedOutput = LabelToOneHotEncoding(labels[n], Layers.back()->outputHeight);

		float loss = CrossEntropyLoss(expectedOutput, Layers.back()->outputs);
		BackPropogate(expectedOutput);
	}
}

void NeuralNetwork::BackPropogate(const std::vector<float>& expected)
{
	for (size_t i = 0; i < expected.size(); i++)
		Layers.back()->outputGradients[i] = Layers.back()->outputs[i] - expected[i];

	for (auto& layer : std::views::reverse(Layers)) {
		layer->BackPropogate();
	}
}

inline void NeuralNetwork::FeedForward()
{
	for (auto& layer : Layers)
	{
		layer->FeedForward();
	}
}
