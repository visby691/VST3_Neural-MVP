#include "PluginProcessor.h"
#include "PluginEditor.h"

NeuralMVPAudioProcessor::NeuralMVPAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

void NeuralMVPAudioProcessor::prepareToPlay(double, int)
{
}

void NeuralMVPAudioProcessor::releaseResources()
{
}

bool NeuralMVPAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& input = layouts.getMainInputChannelSet();
    const auto& output = layouts.getMainOutputChannelSet();

    return input == output
        && (output == juce::AudioChannelSet::mono()
            || output == juce::AudioChannelSet::stereo());
}

void NeuralMVPAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                           juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    juce::ignoreUnused(midiMessages);

    const auto totalInputChannels = getTotalNumInputChannels();
    const auto totalOutputChannels = getTotalNumOutputChannels();

    for (auto channel = totalInputChannels; channel < totalOutputChannels; ++channel)
        buffer.clear(channel, 0, buffer.getNumSamples());
}

juce::AudioProcessorEditor* NeuralMVPAudioProcessor::createEditor()
{
    return new NeuralMVPAudioProcessorEditor(*this);
}

bool NeuralMVPAudioProcessor::hasEditor() const
{
    return true;
}

const juce::String NeuralMVPAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool NeuralMVPAudioProcessor::acceptsMidi() const
{
    return false;
}

bool NeuralMVPAudioProcessor::producesMidi() const
{
    return false;
}

bool NeuralMVPAudioProcessor::isMidiEffect() const
{
    return false;
}

double NeuralMVPAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int NeuralMVPAudioProcessor::getNumPrograms()
{
    return 1;
}

int NeuralMVPAudioProcessor::getCurrentProgram()
{
    return 0;
}

void NeuralMVPAudioProcessor::setCurrentProgram(int)
{
}

const juce::String NeuralMVPAudioProcessor::getProgramName(int)
{
    return {};
}

void NeuralMVPAudioProcessor::changeProgramName(int, const juce::String&)
{
}

void NeuralMVPAudioProcessor::getStateInformation(juce::MemoryBlock&)
{
}

void NeuralMVPAudioProcessor::setStateInformation(const void*, int)
{
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralMVPAudioProcessor();
}
