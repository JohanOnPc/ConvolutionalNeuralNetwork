#pragma once

#include <vector>

void InitWeights(std::vector<float>& weights, size_t amount, size_t fanIn);
void PrintVector(const std::vector<float>& vec);
float CrossEntropyLoss(const std::vector<float>& expected, const std::vector<float>& output);
std::vector<float> LabelToOneHotEncoding(size_t label, size_t outputSize);