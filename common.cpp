#include "common.h"

#include <random>
#include <iostream>

void InitWeights(std::vector<float>& weights, size_t amount, size_t fanIn)
{
	std::random_device rd{};
	std::mt19937 gen{ rd()};

	std::normal_distribution dis{ 0.f, std::sqrtf( 2.f / fanIn )};

	for (size_t i = 0; i < amount; i++)
	{
		weights.push_back(dis(gen));
	}
}

void PrintVector(const std::vector<float>& vec)
{
	std::cout << "[";

	for (const auto& val : vec)
	{
		if (val == vec.back())
			std::cout << val << "]\n";
		else
			std::cout << val << ", ";
	}
}

float CrossEntropyLoss(const std::vector<float>& expected, const std::vector<float>& output)
{
	if (expected.size() != output.size()) {
		std::cout << "Error, the size of the given output and expected output are not the same!\n";
		exit(1);
	}

	float error = 0.f;

	for (size_t i = 0; i < expected.size(); i++) {
		error += expected[i] * std::logf(output[i]);
	}

	return -error;
}

std::vector<float> LabelToOneHotEncoding(size_t label, size_t outputSize)
{

}
