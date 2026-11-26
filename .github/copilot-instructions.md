# Copilot Instructions for Neural Audio Plugin Platform

## Tech Stack & Build Environment
- **Language:** C++20
- **Build System:** CMake (3.22+)
- **Framework:** JUCE 8 (managed via CMake FetchContent or git submodule)
- **Inference Engine:** RTNeural (Header-only / CMake integration, dynamic model loading via `RTNeural::json_parser`)
- **Target Format:** VST3, Standalone (Windows 11 / MSVC)

## Critical Rules: Real-Time Audio Safety
`AudioProcessor::processBlock()` runs on a high-priority real-time audio thread. Violations cause audio dropouts or crashes.
1. **NO Dynamic Memory Allocation:** Never use `new`, `delete`, `malloc`, `std::vector::push_back`, `std::string` manipulation, or `nlohmann::json` parsing inside `processBlock()`.
2. **NO Blocking / Locks:** Never use `std::mutex::lock()` inside `processBlock()`. Use `std::atomic` for simple parameters, or a lock-free pointer swap / `juce::AbstractFifo` when transferring newly loaded models from the Message Thread to the Audio Thread.
3. **NO File I/O or Logging:** Never read files or call `std::cout` / `DBG()` inside `processBlock()`.
4. **Pre-allocation:** Allocate all scratch buffers and reset RTNeural states (`model->reset()`) inside `prepareToPlay()`.

## Coding Guidelines
- Use `juce::AudioProcessorValueTreeState` (APVTS) for all DSP parameters to ensure DAW automation and state saving work out of the box.
- Handle denormal numbers using `juce::ScopedNoDenormals` at the top of `processBlock()`.
- Keep DSP logic decoupled from `PluginProcessor` where possible (e.g., create a dedicated `NeuralEngine` wrapper class).
- Do not invent non-existent JUCE 8 or RTNeural APIs. Stick strictly to documented headers.