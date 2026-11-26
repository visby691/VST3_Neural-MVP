cmake_minimum_required(VERSION 3.22)

project(NeuralMVP VERSION 0.1.0)

include(FetchContent)

FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.6
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(NeuralMVP
    COMPANY_NAME "Visby Audio"
    BUNDLE_ID "com.visbyaudio.neuralmvp"
    PLUGIN_MANUFACTURER_CODE VSBY
    PLUGIN_CODE NMP1
    PRODUCT_NAME "Neural MVP"
    FORMATS VST3 Standalone
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
)

juce_generate_juce_header(NeuralMVP)

target_sources(NeuralMVP
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
)

target_compile_features(NeuralMVP PRIVATE cxx_std_20)

target_compile_definitions(NeuralMVP
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
)

target_link_libraries(NeuralMVP
    PRIVATE
        juce::juce_audio_utils
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
