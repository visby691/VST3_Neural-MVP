# Phase 1 Specification: Minimal Neural Inference VST3 (MVP)

## 1. Objective
Build a minimal VST3 / Standalone audio plugin using JUCE and RTNeural that loads a pre-trained neural network model (RTNeural JSON format) at runtime and processes incoming audio in real time.

## 2. Scope (What to Build in Phase 1)
- **In Scope:**
  - CMake project setup integrating JUCE and RTNeural.
  - Real-time mono/stereo audio inference using `RTNeural::Model<float>`.
  - Thread-safe dynamic loading of `.json` model files via a GUI button.
  - Basic DSP controls: Input Gain, Output Gain, Bypass.
  - Minimal native JUCE GUI (WebView/React will be added in Phase 2).
- **Out of Scope:**
  - Custom `.amlp` zip packaging, WebView UI, Cloud/Firebase integration, Model training scripts.

## 3. Functional Requirements
### 3.1 DSP & Inference (`NeuralEngine`)
- Support dynamic model architecture loading via `RTNeural::json_parser::parseJson()`.
- Process audio sample-by-sample or block-by-block using `model->forward()`.
- **Channel Handling:** Process mono input through the model; for stereo tracks, either sum to mono or instantiate two independent `RTNeural::Model<float>` instances (one per channel) to preserve stereo imaging. (Start with dual-mono: 2 model instances).
- **Parameters (APVTS):**
  - `inputGain`: -24.0 dB to +24.0 dB (default: 0.0 dB)
  - `outputGain`: -24.0 dB to +24.0 dB (default: 0.0 dB)
  - `bypass`: Boolean (default: false)

### 3.2 Thread-Safe Model Loading
1. User clicks "Load Model" button on the GUI (Message Thread).
2. `juce::FileChooser` opens to select a `.json` file.
3. Parse the JSON and construct the new `RTNeural::Model<float>` instance on the Message Thread (or a background thread), calling `reset()` before handoff.
4. Safely swap the active model pointer used by the Audio Thread without locking `processBlock()`. (e.g., double-buffering or `std::atomic` swap with deferred deletion on the Message Thread).
5. If no model is loaded or JSON parsing fails, pass audio through cleanly (unity gain bypass) and display the error status on the GUI.

### 3.3 Minimal GUI (`PluginEditor`)
- A "Load Model (.json)" button.
- A status label showing the currently loaded filename or "No Model Loaded".
- Two rotary sliders for Input Gain and Output Gain, plus a Bypass toggle button, attached to APVTS.

## 4. Suggested Directory Structure
```text
/
├── .github/
│   └── copilot-instructions.md
├── docs/
│   └── phase1_spec.md
├── CMakeLists.txt
└── Source/
    ├── PluginProcessor.h / .cpp
    ├── PluginEditor.h / .cpp
    └── NeuralEngine.h / .cpp