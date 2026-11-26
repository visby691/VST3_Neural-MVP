#pragma once

#include <JuceHeader.h>
#include "NeuralEngine.h"

class NeuralMVPAudioProcessor final : public juce::AudioProcessor,
                                      private juce::Timer,
                                      private juce::AsyncUpdater
{
public:
    NeuralMVPAudioProcessor();
    ~NeuralMVPAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destinationData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState parameters;
    bool loadModelFromFile(const juce::File& modelFile);
    juce::String getModelName() const;
    juce::String getModelStatus() const;
    juce::String getModelError() const;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void timerCallback() override;
    void handleAsyncUpdate() override;

    NeuralEngine neuralEngine;
    std::atomic<float>* inputGainParameter = nullptr;
    std::atomic<float>* outputGainParameter = nullptr;
    std::atomic<float>* bypassParameter = nullptr;
    juce::String modelPath;
    juce::String modelName;
    juce::String modelStatus { "No Model Loaded" };
    juce::String modelError;
    juce::CriticalSection modelStateLock;
    juce::CriticalSection pendingStateLock;
    juce::String pendingStateModelPath;
    bool pendingStateHasModel = false;
};
