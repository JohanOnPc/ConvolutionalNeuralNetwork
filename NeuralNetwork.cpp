#include "NeuralNetwork.h"
#include "common.h"

#include <iostream>
#include <fstream>
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

void NeuralNetwork::Create(float learningRate, float decayRate)
{
	NeuralLayer* previousLayer = nullptr;
	for (auto& layer : Layers)
	{
		layer->Create(previousLayer);
		previousLayer = layer;
		layer->learningRate = learningRate;
	}

	this->learningRate = learningRate;
	this->decayRate = decayRate;
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

void NeuralNetwork::Fit(size_t epochs, const DataSet& dataSet)
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

		std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Learning Rate: " << learningRate << '\n';
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

		std::cout << "  Fitting " << elapsedTime << " - Loss: " << totalLoss / static_cast<float>(trainInput.size()) << " - Accuracy : " << (static_cast<float>(trainCorrect) / static_cast<float>(trainInput.size())) * 100.f << " % - NaNs : " << NaNs << "\n";

		for (size_t n = 0; n < validationInput.size(); n++) {
			auto prediction = Predict(validationInput[n]);
			auto expectedOutput = LabelToOneHotEncoding(validationLabels[n], Layers.back()->outputHeight);

			if (std::distance(Layers.back()->outputs.begin(), std::ranges::max_element(Layers.back()->outputs)) == validationLabels[n])
				validationCorrect++;

			float loss = CrossEntropyLoss(expectedOutput, prediction);

			totalValidationLoss += loss;
		}

		std::cout << "  Validation - Loss: " << totalValidationLoss / static_cast<float>(validationInput.size()) << " - Accuracy : " << (static_cast<float>(validationCorrect) / static_cast<float>(validationInput.size())) * 100.f << " % \n";


		//Update learning rate based on the decay rate

		learningRate /= (1.f + decayRate);
	}
}

void NeuralNetwork::SetLearningRate(float learningRate) const
{
	for (auto& layer : Layers)
		layer->learningRate = learningRate;
}

void NeuralNetwork::SaveModel(const std::string& fileName) const
{
	std::ofstream file(fileName, std::ios::binary | std::fstream::out);

	if (file.is_open()) {
		size_t size =  Layers.size();
		file.write((const char*)&size, sizeof(size));

		for (const auto& layer : Layers)
			layer->SaveLayer(file);
	} 
	else {
		std::cout << "Error, could not create file named: " << fileName;
		exit(1);
	}

	file.close();
}

void NeuralNetwork::LoadModel(const std::string& fileName)
{
	std::ifstream file(fileName, std::ios::binary | std::fstream::in);

	if (file.is_open()) {
		size_t size = 0;
		file.read((char*)&size, sizeof(size));

		for (int i = 0; i < size; i++) {
			uint8_t layerType;
			size_t outputChannels, outputHeight, outputWidth;

			file.read((char*)&layerType, sizeof(layerType));

			file.read((char*)&outputChannels, sizeof(outputChannels));
			file.read((char*)&outputHeight, sizeof(outputHeight));
			file.read((char*)&outputWidth, sizeof(outputWidth));

			switch (layerType)
			{
			case InputLayer:
				this->AddLayer(new Input(outputWidth, outputHeight, outputChannels));
				break;
			case ConvolutionLayer:
				size_t kernelAmount, kernelSize, padding;
				std::string activationFunction;

				file >> activationFunction;

				file.read((char*)&kernelSize, sizeof(kernelSize));
				file.read((char*)&kernelAmount, sizeof(kernelAmount));
				file.read((char*)&padding, sizeof(padding));

				this->AddLayer(new Convolution(kernelAmount, kernelSize, padding, 1, activationFunction));
			}
					
		}
	}
	else {
		std::cout << "Error, could not open file named: " << fileName;
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
