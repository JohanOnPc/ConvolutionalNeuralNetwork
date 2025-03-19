#include "common.h"

#include <random>

void InitWeights(std::vector<float>& weights, size_t amount, size_t fanIn)
{
	std::random_device rd{};
	std::mt19937 gen{ rd()};

	std::normal_distribution dis{ 0.f, 2.f / fanIn };

	for (size_t i = 0; i < amount; i++)
	{
		weights.push_back(dis(gen));
	}
}
