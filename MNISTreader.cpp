#include "MNISTreader.h"

#include <fstream>
#include <iostream>
#include <bit>

DataSet ReadMNISTDataSet(const std::string& trainData, const std::string& trainLabels, const std::string& validationData, const std::string& validationLabels)
{
    auto trainSet = ReadIDXFileData(trainData);
    auto trainLabel = ReadIDXFileLabels(trainLabels);
    auto validationSet = ReadIDXFileData(validationData);
    auto validationLabel = ReadIDXFileLabels(validationLabels);

    return { trainSet, trainLabel, validationSet, validationLabel };
}

std::vector<size_t> ReadIDXFileLabels(const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        std::cout << "Error, could not open the given file: " << fileName << '\n';
        exit(1);
    }

    file.seekg(4);
    uint32_t amount;

    file.read(reinterpret_cast<char*>(&amount), 4);
    amount = std::byteswap(amount);

    std::vector<size_t> labelSet;

    labelSet.reserve(amount);

    uint8_t* buffer = new uint8_t[amount];

    file.read(reinterpret_cast<char*>(buffer), amount);

    for (uint32_t i = 0; i < amount; i++) {
        labelSet.push_back(static_cast<size_t>(buffer[i]));
    }

    delete[] buffer;

    return labelSet;
}

std::vector<std::vector<float>> ReadIDXFileData(const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        std::cout << "Error, could not open the given file: " << fileName << '\n';
        exit(1);
    }

    file.seekg(4);
    uint32_t amount, height, width;

    file.read(reinterpret_cast<char*>(&amount), 4);
    file.read(reinterpret_cast<char*>(&height), 4);
    file.read(reinterpret_cast<char*>(&width), 4);

    amount = std::byteswap(amount);
    height = std::byteswap(height);
    width = std::byteswap(width);

    std::vector<std::vector<float>> imageSet;

    uint8_t* buffer = new uint8_t[width * height];

    for (uint32_t j = 0; j < amount; j++) {
        std::vector<float> image;
        image.reserve(width * height);

        file.read(reinterpret_cast<char*>(buffer), width * height);

        for (uint32_t i = 0; i < width * height; i++) {      
            image.push_back(static_cast<float>(buffer[i]) / 255.f);
        }

        imageSet.push_back(image);
    }

    delete[] buffer;

    return imageSet;
}
