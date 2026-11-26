#pragma once

#include <JuceHeader.h>

class NeuralMVPAudioProcessor;

class NeuralMVPAudioProcessorEditor final : public juce::AudioProcessorEditor
{
public:
    explicit NeuralMVPAudioProcessorEditor(NeuralMVPAudioProcessor& processor);

    void paint(juce::Graphics& graphics) override;
    void resized() override;
};
