#include "PluginProcessor.h"
#include "PluginEditor.h"

NeuralMVPAudioProcessor::NeuralMVPAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, "PARAMETERS", createParameterLayout())
{
    inputGainParameter = parameters.getRawParameterValue("inputGain");
    outputGainParameter = parameters.getRawParameterValue("outputGain");
    bypassParameter = parameters.getRawParameterValue("bypass");
    startTimerHz(2);
}

NeuralMVPAudioProcessor::~NeuralMVPAudioProcessor()
{
    stopTimer();
    cancelPendingUpdate();
}

void NeuralMVPAudioProcessor::prepareToPlay(double, int)
{
    neuralEngine.requestReset();
}

void NeuralMVPAudioProcessor::releaseResources()
{
    neuralEngine.requestReset();
}

juce::AudioProcessorValueTreeState::ParameterLayout
NeuralMVPAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "inputGain", 1 }, "Input Gain",
        juce::NormalisableRange<float>(-24.0f, 24.0f, 0.1f), 0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "outputGain", 1 }, "Output Gain",
        juce::NormalisableRange<float>(-24.0f, 24.0f, 0.1f), 0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { "bypass", 1 }, "Bypass", false));
    return layout;
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
    const auto sampleCount = buffer.getNumSamples();
    const auto inputGain = juce::Decibels::decibelsToGain(
        inputGainParameter->load(std::memory_order_relaxed));
    const auto outputGain = juce::Decibels::decibelsToGain(
        outputGainParameter->load(std::memory_order_relaxed));

    for (auto channel = 0; channel < totalInputChannels; ++channel)
        buffer.applyGain(channel, 0, sampleCount, inputGain);

    if (bypassParameter->load(std::memory_order_relaxed) < 0.5f)
        neuralEngine.process(buffer);

    for (auto channel = 0; channel < totalInputChannels; ++channel)
        buffer.applyGain(channel, 0, sampleCount, outputGain);

    for (auto channel = totalInputChannels; channel < totalOutputChannels; ++channel)
        buffer.clear(channel, 0, sampleCount);
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

void NeuralMVPAudioProcessor::getStateInformation(juce::MemoryBlock& destinationData)
{
    auto state = parameters.copyState();
    juce::String savedModelPath;
    {
        const juce::ScopedLock lock(modelStateLock);
        savedModelPath = modelPath;
    }
    state.setProperty("modelPath", savedModelPath, nullptr);
    if (auto xml = state.createXml())
        copyXmlToBinary(*xml, destinationData);
}

void NeuralMVPAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary(data, sizeInBytes))
    {
        auto restoredState = juce::ValueTree::fromXml(*xml);
        if (!restoredState.isValid() || restoredState.getType() != parameters.state.getType())
            return;

        const auto restoredModelPath = restoredState.getProperty("modelPath").toString();
        parameters.replaceState(restoredState);

        {
            const juce::ScopedLock lock(pendingStateLock);
            pendingStateModelPath = restoredModelPath;
            pendingStateHasModel = restoredModelPath.isNotEmpty();
        }
        triggerAsyncUpdate();
    }
}

bool NeuralMVPAudioProcessor::loadModelFromFile(const juce::File& file)
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    juce::String errorMessage;
    if (!neuralEngine.loadModel(file, errorMessage))
    {
        const juce::ScopedLock lock(modelStateLock);
        modelError = errorMessage;
        modelStatus = "Load Failed";
        return false;
    }

    {
        const juce::ScopedLock lock(modelStateLock);
        modelPath = file.getFullPathName();
        modelName = file.getFileName();
        modelStatus = "Loaded";
        modelError.clear();
    }
    neuralEngine.requestReset();
    return true;
}

juce::String NeuralMVPAudioProcessor::getModelName() const
{
    const juce::ScopedLock lock(modelStateLock);
    return modelName.isNotEmpty() ? modelName : "No Model Loaded (bypassed)";
}

juce::String NeuralMVPAudioProcessor::getModelStatus() const
{
    const juce::ScopedLock lock(modelStateLock);
    return modelStatus;
}

juce::String NeuralMVPAudioProcessor::getModelError() const
{
    const juce::ScopedLock lock(modelStateLock);
    return modelError;
}

void NeuralMVPAudioProcessor::timerCallback()
{
    neuralEngine.collectRetiredModels();
}

void NeuralMVPAudioProcessor::handleAsyncUpdate()
{
    juce::String restoredModelPath;
    bool hasRestoredModel = false;
    {
        const juce::ScopedLock lock(pendingStateLock);
        restoredModelPath = pendingStateModelPath;
        hasRestoredModel = pendingStateHasModel;
    }

    if (hasRestoredModel)
    {
        loadModelFromFile(juce::File(restoredModelPath));
        return;
    }

    neuralEngine.unloadModel();
    {
        const juce::ScopedLock lock(modelStateLock);
        modelPath.clear();
        modelName.clear();
        modelStatus = "No Model Loaded";
        modelError.clear();
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralMVPAudioProcessor();
}
