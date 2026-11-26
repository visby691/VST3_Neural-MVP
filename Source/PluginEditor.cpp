#include "PluginEditor.h"
#include "PluginProcessor.h"

NeuralMVPAudioProcessorEditor::NeuralMVPAudioProcessorEditor(
    NeuralMVPAudioProcessor& processor)
    : AudioProcessorEditor(processor),
      audioProcessor(processor),
      inputGainAttachment(processor.parameters, "inputGain", inputGainSlider),
      outputGainAttachment(processor.parameters, "outputGain", outputGainSlider),
      bypassAttachment(processor.parameters, "bypass", bypassButton)
{
    loadModelButton.onClick = [this] { chooseModel(); };
    addAndMakeVisible(loadModelButton);

    for (auto* label : { &modelNameLabel, &modelStatusLabel, &modelErrorLabel })
    {
        label->setJustificationType(juce::Justification::centredLeft);
        addAndMakeVisible(*label);
    }

    modelErrorLabel.setColour(juce::Label::textColourId, juce::Colours::salmon);
    inputGainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    inputGainSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 20);
    inputGainSlider.setTextValueSuffix(" dB");
    outputGainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    outputGainSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 20);
    outputGainSlider.setTextValueSuffix(" dB");
    addAndMakeVisible(inputGainSlider);
    addAndMakeVisible(outputGainSlider);
    addAndMakeVisible(bypassButton);

    refreshModelStatus();
    startTimerHz(4);
    setSize(480, 400);
}

void NeuralMVPAudioProcessorEditor::paint(juce::Graphics& graphics)
{
    graphics.fillAll(juce::Colour(0xff20242a));
    graphics.setColour(juce::Colours::white);
    graphics.setFont(24.0f);
    graphics.drawText("Neural MVP", 16, 8, getWidth() - 32, 32,
                      juce::Justification::centred);
}

void NeuralMVPAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced(16);
    area.removeFromTop(32);
    loadModelButton.setBounds(area.removeFromTop(36));
    area.removeFromTop(8);
    modelNameLabel.setBounds(area.removeFromTop(24));
    modelStatusLabel.setBounds(area.removeFromTop(22));
    modelErrorLabel.setBounds(area.removeFromTop(36));
    area.removeFromTop(8);

    auto controls = area.removeFromTop(170);
    const auto sliderWidth = controls.getWidth() / 2;
    inputGainSlider.setBounds(controls.removeFromLeft(sliderWidth));
    outputGainSlider.setBounds(controls.removeFromLeft(sliderWidth));
    bypassButton.setBounds(area.removeFromTop(28));
}

void NeuralMVPAudioProcessorEditor::chooseModel()
{
    fileChooser = std::make_unique<juce::FileChooser>(
        "Select an RTNeural model", juce::File(), "*.json");
    juce::Component::SafePointer<NeuralMVPAudioProcessorEditor> safeThis(this);
    fileChooser->launchAsync(juce::FileBrowserComponent::openMode
                                 | juce::FileBrowserComponent::canSelectFiles,
                             [safeThis](const juce::FileChooser& chooser)
                             {
                                 if (safeThis == nullptr)
                                     return;

                                 const auto selectedFile = chooser.getResult();
                                 if (selectedFile.existsAsFile())
                                     safeThis->audioProcessor.loadModelFromFile(selectedFile);
                                 safeThis->refreshModelStatus();
                             });
}

void NeuralMVPAudioProcessorEditor::refreshModelStatus()
{
    modelNameLabel.setText("Model: " + audioProcessor.getModelName(),
                           juce::dontSendNotification);
    modelStatusLabel.setText("Status: " + audioProcessor.getModelStatus(),
                             juce::dontSendNotification);
    modelErrorLabel.setText(audioProcessor.getModelError(),
                            juce::dontSendNotification);
}

void NeuralMVPAudioProcessorEditor::timerCallback()
{
    refreshModelStatus();
}
