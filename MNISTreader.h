#pragma once

#include "common.h"

#include <vector>
#include <string>


DataSet ReadMNISTDataSet(const std::string& trainData, const std::string& trainLabels, const std::string& validationData, const std::string& validationLabels);

std::vector<size_t> ReadIDXFileLabels(const std::string& fileName);
std::vector<std::vector<float>> ReadIDXFileData(const std::string& fileName);