#include "NeuralEngine.h"

#include <stdexcept>
#include <string>

namespace
{
void validateModelJson(const nlohmann::json& json)
{
    const auto& layers = json.at("layers");
    if (!layers.is_array() || layers.empty())
        throw std::runtime_error("The JSON does not contain any model layers.");

    for (const auto& layer : layers)
    {
        if (!layer.is_object() || !layer.contains("type") || !layer["type"].is_string())
            throw std::runtime_error("A model layer is missing its type.");

        const auto type = layer["type"].get<std::string>();
        if (type != "dense" && type != "time-distributed-dense"
            && type != "conv1d" && type != "conv2d" && type != "gru"
            && type != "lstm" && type != "prelu" && type != "batchnorm"
            && type != "batchnorm2d" && type != "activation")
        {
            throw std::runtime_error("The model contains an unsupported layer type: " + type);
        }

        if (type == "activation" && !layer.contains("activation"))
            throw std::runtime_error("An activation layer is missing its activation type.");

        if (layer.contains("activation"))
        {
            if (!layer["activation"].is_string())
                throw std::runtime_error("A layer has an invalid activation type.");

            const auto activation = layer["activation"].get<std::string>();
            if (type == "activation" && activation.empty())
                throw std::runtime_error("An activation layer must specify an activation type.");

            if (!activation.empty() && activation != "tanh" && activation != "relu"
                && activation != "sigmoid" && activation != "softmax" && activation != "elu")
            {
                throw std::runtime_error("The model contains an unsupported activation type: " + activation);
            }
        }
    }
}

std::unique_ptr<RTNeural::Model<float>> createModel(const nlohmann::json& json)
{
    auto model = RTNeural::json_parser::parseJson<float>(json);

    if (model == nullptr || model->layers.empty())
        throw std::runtime_error("The JSON does not contain a usable neural network.");

    for (const auto* layer : model->layers)
    {
        if (layer == nullptr)
            throw std::runtime_error("The model contains an unsupported or invalid layer.");
    }

    if (model->getInSize() != 1 || model->getOutSize() != 1)
        throw std::runtime_error("The model must have exactly one input and one output.");

    model->reset();
    return model;
}
}

NeuralEngine::NeuralEngine() = default;

NeuralEngine::~NeuralEngine()
{
    delete pendingModel.exchange(nullptr, std::memory_order_acq_rel);
    delete activeModel;
    collectRetiredModels();
}

bool NeuralEngine::loadModel(const juce::File& modelFile, juce::String& errorMessage)
{
    errorMessage.clear();

    if (!modelFile.hasFileExtension(".json"))
    {
        errorMessage = "The selected file must have a .json extension.";
        return false;
    }

    if (!modelFile.existsAsFile())
    {
        errorMessage = "The selected model file does not exist.";
        return false;
    }

    juce::MemoryBlock fileContents;
    if (!modelFile.loadFileAsData(fileContents) || fileContents.getSize() == 0)
    {
        errorMessage = "The model file is empty or could not be read.";
        return false;
    }

    try
    {
        const auto* data = static_cast<const char*>(fileContents.getData());
        const std::string jsonText(data, fileContents.getSize());
        const auto json = nlohmann::json::parse(jsonText);

        if (!json.is_object())
            throw std::runtime_error("The model JSON root must be an object.");

        validateModelJson(json);

        auto newModels = std::make_unique<ModelPair>();
        newModels->left = createModel(json);
        newModels->right = createModel(json);

        collectRetiredModels();

        auto* replacement = newModels.release();
        delete pendingModel.exchange(replacement, std::memory_order_acq_rel);
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = juce::String("Could not load model: ") + exception.what();
        return false;
    }
}

void NeuralEngine::unloadModel()
{
    auto* unloadedModel = new ModelPair();
    delete pendingModel.exchange(unloadedModel, std::memory_order_acq_rel);
}

void NeuralEngine::collectRetiredModels()
{
    int start1 = 0;
    int size1 = 0;
    int start2 = 0;
    int size2 = 0;
    retirementFifo.prepareToRead(retirementQueueSize, start1, size1, start2, size2);

    for (auto index = 0; index < size1; ++index)
        delete retiredModels[static_cast<size_t>(start1 + index)];

    for (auto index = 0; index < size2; ++index)
        delete retiredModels[static_cast<size_t>(start2 + index)];

    retirementFifo.finishedRead(size1 + size2);
}

void NeuralEngine::process(juce::AudioBuffer<float>& audioBuffer) noexcept
{
    if (resetRequested.exchange(false, std::memory_order_acq_rel) && activeModel != nullptr)
    {
        if (activeModel->left != nullptr)
            activeModel->left->reset();
        if (activeModel->right != nullptr)
            activeModel->right->reset();
    }

    if (pendingModel.load(std::memory_order_acquire) != nullptr)
    {
        if (activeModel == nullptr)
        {
            activeModel = pendingModel.exchange(nullptr, std::memory_order_acq_rel);
        }
        else
        {
            int start1 = 0;
            int size1 = 0;
            int start2 = 0;
            int size2 = 0;

            if (retirementFifo.getFreeSpace() > 0)
            {
                retirementFifo.prepareToWrite(1, start1, size1, start2, size2);
                if (size1 > 0 || size2 > 0)
                {
                    if (auto* replacement = pendingModel.exchange(nullptr, std::memory_order_acq_rel))
                    {
                        const auto writeIndex = size1 > 0 ? start1 : start2;
                        retiredModels[static_cast<size_t>(writeIndex)] = activeModel;
                        retirementFifo.finishedWrite(1);
                        activeModel = replacement;
                    }
                    else
                    {
                        retirementFifo.finishedWrite(0);
                    }
                }
            }
        }
    }

    if (activeModel == nullptr || activeModel->left == nullptr || activeModel->right == nullptr)
        return;

    const auto channelCount = juce::jmin(audioBuffer.getNumChannels(), 2);
    const auto sampleCount = audioBuffer.getNumSamples();

    for (auto channel = 0; channel < channelCount; ++channel)
    {
        auto* samples = audioBuffer.getWritePointer(channel);
        auto* model = (channel == 0 ? activeModel->left : activeModel->right).get();

        for (auto sample = 0; sample < sampleCount; ++sample)
            samples[sample] = model->forward(&samples[sample]);
    }
}

void NeuralEngine::requestReset() noexcept
{
    resetRequested.store(true, std::memory_order_release);
}
