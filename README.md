# Promptly: Intelligent Coding Assistant ![icon3](https://github.com/user-attachments/assets/838b719b-b863-4310-8a5f-be140960d230)


Promptly is a powerful Visual Studio Code extension that integrates advanced AI capabilities into your coding workflow. It supports multiple AI models including Gemini AI, OpenAI's GPT, Anthropic's Claude, and local language models, providing intelligent assistance for various programming tasks.

## Features
________________
### AI-Powered Coding Assistance
- **AI-Powered Chat**: Engage in context-aware conversations about your code.
- **Code Generation**: Generate code snippets based on natural language descriptions.
- **Prompt Mode**: Quickly enter prompts with a dedicated marker for efficient interactions.
- **Command Mode**: Execute specific commands like checking server status or shutting down local models.

### Multi-Model and Local Support
- **Multi-Model Support**: Choose between Gemini, GPT, Claude, or use local models like Llama 3 and Mistral.
- **Local Model Support**: Set up and use local language models for offline capabilities.

### Jupyter Notebook Integration
- **Seamless Assistance**: AI-powered features are fully integrated within Jupyter notebooks.
- **Automatic Error Detection**: Promptly can automatically detect and offer assistance for traceback errors.

### Customization and User Experience
- **Customizable Keybindings**: Quick access to AI features with customizable shortcuts.
- **Language-Specific Comments**: Automatically uses the correct comment syntax for different programming languages.

## Installation
________________
1. Open Visual Studio Code
2. Go to the Extensions view (Ctrl+Shift+X)
3. Search for "Promptly"
4. Click Install


## Quick Start
________________
1. After installation, open the Command Palette (Ctrl+Shift+P)
2. Run "Promptly: Switch Model" to select your preferred AI model
3. If using an API-based model, set up your API key in the settings
4. For local models, run "Promptly: Setup Local Model" and follow the prompts
5. Start chatting with Ctrl+Shift+Z (Cmd+Shift+Z on Mac) or use the Command Palette



## Setup
________________
### API-based Models (Gemini, GPT, Claude)
1. Obtain an API key from the respective provider:
   - [Google AI Studio](https://makersuite.google.com/app/apikey) for Gemini
   - [OpenAI](https://platform.openai.com/account/api-keys) for GPT
   - [Anthropic](https://www.anthropic.com/) for Claude
2. In VS Code, go to Settings (File > Preferences > Settings)
3. Search for "Promptly"
4. Enter your API key in the corresponding field (e.g., "Promptly: Gemini Api Key")

### Local Model Setup
1. Run the command "Promptly: Setup Local Model" from the Command Palette
2. Follow the prompts to select and configure your local model
3. Ensure you have the necessary dependencies installed for local model support

## Usage
________________
- Start a Chat: Use Ctrl+Shift+Z (Cmd+Shift+Z on Mac) or run "Promptly: Start Chat" from the Command Palette.
   - Note: If you make a selection before starting a chat, the model will use the selected text as the prompt. If no text is selected, the entire document will be used.
- Extract code from AI response: Use Ctrl+Shift+Q (Cmd+Shift+Q on Mac) or run "Promptly: Extract Code from Last Response"
- Enter prompt mode: Use Shift+. or run "Promptly: Enter Prompt Mode"
   - Note: Only the input entered after the prompt marker will be sent to the model.
- Switch AI model: Run "Promptly: Switch Model" from the Command Palette
- Setup local model: Run "Promptly: Setup Local Model" from the Command Palette
- Enter command mode: Use the "?" character to execute specific commands

## Command Mode

In command mode, you can execute specific actions:

- status server: Check the status of the local model server
- shutdown server: Safely shut down the local model server

## Configuration
________________
Promptly offers several configuration options to customize its behavior:

### API Keys
To use Promptly with different AI models, you need to configure the appropriate API keys:

- `promptly.geminiApiKey`: API Key for Gemini AI
- `promptly.openaiApiKey`: API Key for OpenAI
- `promptly.anthropicApiKey`: API Key for Anthropic

You can set these in your VS Code settings.

⸻⸻⸻⸻
### Custom Prompts
Promptly allows you to create and use custom prompts to tailor the AI's behavior:

1. Open VS Code Settings (File > Preferences > Settings)
2. Search for "promptly custom prompts"
3. In the "Promptly: Custom Prompts" section, click "Add Item"
4. Enter a name for your prompt in the "Item" field
5. Enter your custom prompt text in the "Value" field; click OK when finished.

To select an active prompt:
1. Click the "Prompt: [current prompt]" item in the status bar (bottom-right)
2. Choose your desired prompt from the list

The selected prompt will be used for all AI interactions until changed.

⸻⸻⸻⸻
### Custom Hotkeys
Promptly supports user-defined keybindings for its main commands. You can easily configure these in your VS Code settings:

- `promptly.hotkey.startChat`: Start an AI chat session
- `promptly.hotkey.extractCode`: Extract code from the last AI response
- `promptly.hotkey.enterPromptMode`: Enter prompt mode
- `promptly.hotkey.switchModel`: Switch between AI models
- `promptly.hotkey.handleTracebackError`: Handle traceback errors
- `promptly.hotkey.setupLocalModel`: Setup a local AI model

To set or change a hotkey:

1. Open VS Code Settings (File > Preferences > Settings, or `Cmd+,` on Mac, `Ctrl+,` on Windows/Linux)
2. Search for "promptly hotkey"
3. Find the Promptly: Hotkeys section
4. Select the command you want to set a hotkey for by clicking add item
5. Enter the desired key combination in the value field
6. The new hotkey will be automatically saved (update when prompted)

Example:
To set `Ctrl+Shift+A` for starting a chat session, you would set:
```json
"startChat": "ctrl+shift+a"
```



## Troubleshooting
________________
- If you're having issues with API-based models, ensure your API key is correctly set in the settings.
- For local models, make sure you have the necessary dependencies installed and the model path is correctly set.
- If you encounter any errors, check the Output panel (View > Output) and select "Promptly" from the dropdown for detailed logs.

## Feedback and Support
________________
We welcome your feedback and bug reports! Please visit our [GitHub repository](https://github.com/Di-Gi/promptly) to:

- Report issues
- Suggest new features
- Contribute to the project

## Requirements
________________
- Visual Studio Code v1.92.0 or higher
- Python extension for VS Code (for local model support)
- Internet connection (for API-based models)

## License
________________
This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/Di-Gi/promptly/blob/main/LICENSE) file for details.
