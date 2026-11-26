#pragma once

#include <JuceHeader.h>

class NeuralMVPAudioProcessor;

class NeuralMVPAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                            private juce::Timer
{
public:
    explicit NeuralMVPAudioProcessorEditor(NeuralMVPAudioProcessor& processor);

    void paint(juce::Graphics& graphics) override;
    void resized() override;

private:
    void timerCallback() override;
    void chooseModel();
    void refreshModelStatus();

    NeuralMVPAudioProcessor& audioProcessor;
    juce::TextButton loadModelButton { "Load Model (.json)" };
    juce::Label modelNameLabel;
    juce::Label modelStatusLabel;
    juce::Label modelErrorLabel;
    juce::Slider inputGainSlider;
    juce::Slider outputGainSlider;
    juce::ToggleButton bypassButton { "Bypass" };
    juce::AudioProcessorValueTreeState::SliderAttachment inputGainAttachment;
    juce::AudioProcessorValueTreeState::SliderAttachment outputGainAttachment;
    juce::AudioProcessorValueTreeState::ButtonAttachment bypassAttachment;
    std::unique_ptr<juce::FileChooser> fileChooser;
};
