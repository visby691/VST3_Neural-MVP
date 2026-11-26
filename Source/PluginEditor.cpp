#include "PluginEditor.h"
#include "PluginProcessor.h"

NeuralMVPAudioProcessorEditor::NeuralMVPAudioProcessorEditor(
    NeuralMVPAudioProcessor& processor)
    : AudioProcessorEditor(processor)
{
    setSize(480, 300);
}

void NeuralMVPAudioProcessorEditor::paint(juce::Graphics& graphics)
{
    graphics.fillAll(juce::Colour(0xff20242a));
    graphics.setColour(juce::Colours::white);
    graphics.setFont(24.0f);
    graphics.drawFittedText("Neural MVP", getLocalBounds(),
                            juce::Justification::centred, 1);
}

void NeuralMVPAudioProcessorEditor::resized()
{
}
