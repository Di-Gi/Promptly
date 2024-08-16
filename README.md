# Promptly: Intelligent Coding Assistant

Promptly is a powerful Visual Studio Code extension that integrates advanced AI capabilities into your coding workflow. It supports multiple AI models including Gemini AI, OpenAI's GPT, Anthropic's Claude, and local language models, providing intelligent assistance for various programming tasks.

## Features

- **AI-Powered Chat**: Engage in context-aware conversations about your code.
- **Code Generation**: Generate code snippets based on natural language descriptions.
- **Error Handling**: Get assistance with error messages and debugging.
- **Multi-Model Support**: Choose between Gemini, GPT, Claude, or use local models like Llama 3 and Mistral.
- **Jupyter Notebook Integration**: Seamless AI assistance within Jupyter notebooks.
- **Customizable Keybindings**: Quick access to AI features with customizable shortcuts.

## Installation

1. Open Visual Studio Code
2. Go to the Extensions view (Ctrl+Shift+X)
3. Search for "Promptly"
4. Click Install

## Quick Start

1. After installation, open the Command Palette (Ctrl+Shift+P)
2. Run "AI Chat: Switch Model" to select your preferred AI model
3. If using an API-based model, set up your API key in the settings
4. For local models, run "AI Chat: Setup Local Model" and follow the prompts
5. Start chatting with Ctrl+Shift+Z (Cmd+Shift+Z on Mac) or use the Command Palette

## Setup

### API-based Models (Gemini, GPT, Claude)
1. Obtain an API key from the respective provider:
   - [Google AI Studio](https://makersuite.google.com/app/apikey) for Gemini
   - [OpenAI](https://platform.openai.com/account/api-keys) for GPT
   - [Anthropic](https://www.anthropic.com/) for Claude
2. In VS Code, go to Settings (File > Preferences > Settings)
3. Search for "Promptly"
4. Enter your API key in the corresponding field (e.g., "Promptly: Gemini Api Key")

### Local Model Setup
1. Run the command "AI Chat: Setup Local Model" from the Command Palette
2. Follow the prompts to select and configure your local model (Llama 3 or Mistral)

## Usage

- Start a chat: Use Ctrl+Shift+Z (Cmd+Shift+Z on Mac) or run "AI Chat: Start Chat" from the Command Palette
- Extract code from AI response: Use Ctrl+Shift+Q (Cmd+Shift+Q on Mac) or run "AI Chat: Extract Code from Last Response"
- Enter prompt mode: Use Shift+. or run "AI Chat: Enter Prompt Mode"
- Switch AI model: Run "AI Chat: Switch Model" from the Command Palette
- Handle errors in Jupyter notebooks: Run "AI Chat: Handle Traceback Error" when an error occurs

## Configuration

Customize Promptly in your VS Code settings:

- `Promptly.geminiApiKey`: Your Gemini AI API key
- `Promptly.openaiApiKey`: Your OpenAI API key
- `Promptly.anthropicApiKey`: Your Anthropic API key
- `Promptly.model`: Choose between available AI models
- `Promptly.localModelPath`: Path to your locally installed model
- `Promptly.localPreconfiguredModel`: Select a pre-configured local model (Llama 3 or Mistral)

## Troubleshooting

- If you're having issues with API-based models, ensure your API key is correctly set in the settings.
- For local models, make sure you have the necessary dependencies installed and the model path is correctly set.
- If you encounter any errors, check the Output panel (View > Output) and select "Promptly" from the dropdown for detailed logs.

## Feedback and Support

We welcome your feedback and bug reports! Please visit our [GitHub repository](https://github.com/Di-GI/promptly) to:

- Report issues
- Suggest new features
- Contribute to the project

## Changelog

See our [CHANGELOG.md](https://github.com/Di-GI/promptly/blob/main/CHANGELOG.md) for a detailed history of changes.

## Requirements

- Visual Studio Code v1.92.0 or higher
- Python extension for VS Code (for local model support)
- Internet connection (for API-based models)

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/Di-GI/promptly/blob/main/LICENSE) file for details.
