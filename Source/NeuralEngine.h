#pragma once

#include <JuceHeader.h>
#include <RTNeural/RTNeural.h>

#include <array>
#include <atomic>
#include <memory>

class NeuralEngine final
{
public:
    NeuralEngine();
    ~NeuralEngine();

    NeuralEngine(const NeuralEngine&) = delete;
    NeuralEngine& operator=(const NeuralEngine&) = delete;

    // Call from the Message Thread. A failed load leaves the current model intact.
    bool loadModel(const juce::File& modelFile, juce::String& errorMessage);

    // Call periodically from a non-audio thread to release replaced models.
    void collectRetiredModels();

    // Processes mono or stereo audio in place; without a loaded model it is a bypass.
    void process(juce::AudioBuffer<float>& audioBuffer) noexcept;

private:
    struct ModelPair
    {
        std::unique_ptr<RTNeural::Model<float>> left;
        std::unique_ptr<RTNeural::Model<float>> right;
    };

    static_assert(std::atomic<ModelPair*>::is_always_lock_free,
                  "NeuralEngine requires lock-free model pointer operations.");

    static constexpr int retirementQueueSize = 8;

    juce::AbstractFifo retirementFifo { retirementQueueSize };
    std::array<ModelPair*, retirementQueueSize> retiredModels {};
    std::atomic<ModelPair*> pendingModel { nullptr };

    // Owned and accessed only by the audio thread, except after audio has stopped
    // during destruction.
    ModelPair* activeModel = nullptr;
};
