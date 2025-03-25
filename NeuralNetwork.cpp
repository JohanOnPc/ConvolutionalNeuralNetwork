#include "NeuralNetwork.h"
#include "common.h"

#include <iostream>
#include <ranges>
#include <chrono>

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

void NeuralNetwork::Fit(size_t epochs, const dataSet& dataSet)
{
	Fit(epochs, dataSet.trainInput, dataSet.trainLabels, dataSet.validationInput, dataSet.validationLabels);
}

void NeuralNetwork::Fit(size_t epochs, const std::vector<std::vector<float>>& trainInput, const std::vector<size_t>& trainLabels, const std::vector<std::vector<float>>& validationInput, const std::vector<size_t>& validationLabels)
{
	//set input to data
	//feed forward through all the layers
	//Calculate the error
	//Propogate the gradients throughout the network

	if (trainInput.size() != trainLabels.size()) {
		std::cout << "Error Fit(), Amount of training inputs is not the same as the amount of labels\n";
		exit(1);
	}

	if (validationInput.size() != validationLabels.size()) {
		std::cout << "Error Fit(), Amount of validation inputs is not the same as the amount of labels\n";
		exit(1);
	}

	for (size_t epoch = 0; epoch < epochs; epoch++) {
		float totalLoss = 0.f, totalValidationLoss = 0.f;
		size_t NaNs = 0;

		size_t trainCorrect = 0, validationCorrect = 0;

		std::cout << "Epoch " << epoch + 1 << "/" << epochs << '\n';
		const auto startTime = std::chrono::steady_clock::now();

		for (size_t n = 0; n < trainInput.size(); n++) {
			Layers.front()->outputs = trainInput[n];
			FeedForward();

			auto expectedOutput = LabelToOneHotEncoding(trainLabels[n], Layers.back()->outputHeight);

			if (std::distance(Layers.back()->outputs.begin(), std::ranges::max_element(Layers.back()->outputs)) == trainLabels[n])
				trainCorrect++;

			float loss = CrossEntropyLoss(expectedOutput, Layers.back()->outputs);
			if (!std::isnan(loss)) {
				BackPropogate(expectedOutput);

				totalLoss += loss;
			}
			else
				NaNs++;
		}

		const auto endTime = std::chrono::steady_clock::now();
		const std::chrono::duration<double> elapsedTime = endTime - startTime;

		std::cout << "  Fitting " << elapsedTime << " - Loss: " << totalLoss / static_cast<float>(trainInput.size()) << " - Accuracy : " << (static_cast<float>(trainCorrect) / static_cast<float>(trainInput.size())) * 100.f << " % -NaNs : " << NaNs << "\n";

		for (size_t n = 0; n < validationInput.size(); n++) {
			auto prediction = Predict(validationInput[n]);
			auto expectedOutput = LabelToOneHotEncoding(validationLabels[n], Layers.back()->outputHeight);

			if (std::distance(Layers.back()->outputs.begin(), std::ranges::max_element(Layers.back()->outputs)) == validationLabels[n])
				validationCorrect++;

			float loss = CrossEntropyLoss(expectedOutput, prediction);

			totalValidationLoss += loss;
		}

		std::cout << "  Validation - Loss: " << totalValidationLoss / static_cast<float>(validationInput.size()) << " - Accuracy : " << (static_cast<float>(validationCorrect) / static_cast<float>(validationInput.size())) * 100.f << " % \n";

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
