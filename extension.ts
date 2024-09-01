import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { window, commands, workspace, ExtensionContext, ConfigurationTarget } from 'vscode';
import { startLocalModelSetup } from './localModelSetup';
import {
    startup, updateModelList, getAvailableModels, handleChat, extractCodeFromLastResponse,
    handleChatNotebook, handleTextDocumentChange, handleNotebookDocumentChange,
    highlightCommandMarker, highlightPromptMarker, handlePromptEnter,
    handleTracebackError
} from './chatUtils';

const PROMPT_MARKER = '>>';

let isPromptMode = false;
let isCommandMode = false;



export async function activate(context: ExtensionContext) {
    try {
        await updateModelList();
        registerCommands(context);
        startup();

        context.subscriptions.push(
            workspace.onDidChangeConfiguration(event => {
                if (event.affectsConfiguration('promptly.hotkeys')) {
                    vscode.window.showInformationMessage(
                        'Promptly: Hotkey configuration has changed. Please run the "Promptly: Update Keybindings" command to apply changes.',
                        'Update Now'
                    ).then(selection => {
                        if (selection === 'Update Now') {
                            vscode.commands.executeCommand('promptly.updateKeybindings');
                        }
                    });
                }
            })
        );
    } catch (error) {
        console.error('Promptly: Error during activation', error);
    }
}

function registerCommands(context: ExtensionContext) {
    const commandHandlers = {
        startChat: handleChat,
        extractCode: extractCodeFromLastResponse,
        startChatNotebook: handleChatNotebook,
        setupLocalModel: startLocalModelSetup,
        handleTracebackError: handleTracebackError,
        enterCommandMode: () => enterMode('command'),
        enterPromptMode: () => enterMode('prompt'),
        switchModel: switchModel,
        updateKeybindings: updateKeybindings
    };

    Object.entries(commandHandlers).forEach(([command, handler]) => {
        const disposable = vscode.commands.registerCommand(`promptly.${command}`, (...args: any[]) => {
            if (typeof handler === 'function') {
                if (handler.length > 0) {
                    return (handler as (...args: any[]) => any)(...args);
                } else {
                    return (handler as () => any)();
                }
            } else {
                console.error(`Promptly: Handler for ${command} is not a function`);
            }
        });
        context.subscriptions.push(disposable);
    });

    context.subscriptions.push(
        vscode.commands.registerCommand('type', handleTypeCommand),
        workspace.onDidChangeTextDocument(handleTextDocumentChange),
        workspace.onDidChangeNotebookDocument(handleNotebookDocumentChange)
    );
}

async function updateKeybindings() {
    const config = workspace.getConfiguration('promptly');
    const hotkeys = config.get('hotkeys') as { [key: string]: string };

    // Update VS Code settings
    await config.update('hotkeys', hotkeys, ConfigurationTarget.Global);

    // Update keybindings.json
    const keybindingsPath = path.join(process.env.APPDATA || '', 'Code', 'User', 'keybindings.json');
    let keybindings: any[] = [];

    if (fs.existsSync(keybindingsPath)) {
        const keybindingsContent = fs.readFileSync(keybindingsPath, 'utf8');
        try {
            // Remove comments before parsing
            const jsonContent = keybindingsContent.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
            keybindings = JSON.parse(jsonContent);
        } catch (error) {
            console.error('Promptly: Error parsing keybindings.json:', error);
            vscode.window.showErrorMessage('Error updating keybindings. Please check your keybindings.json file.');
            return;
        }
    }

    // Remove existing Promptly keybindings
    keybindings = keybindings.filter(kb => !kb.command.startsWith('promptly.'));

    // Add new Promptly keybindings
    for (const [command, key] of Object.entries(hotkeys)) {
        if (key) {
            keybindings.push({
                key: key,
                command: `promptly.${command}`,
                when: 'editorTextFocus'
            });
        }
    }

    // Write updated keybindings back to file
    try {
        fs.writeFileSync(keybindingsPath, JSON.stringify(keybindings, null, 2));
        vscode.window.showInformationMessage('Promptly: Keybindings have been updated successfully.');
    } catch (error) {
        console.error('Promptly: Error writing keybindings.json:', error);
        vscode.window.showErrorMessage('Error saving keybindings. Please check your keybindings.json file permissions.');
    }
}

async function enterMode(mode: 'command' | 'prompt') {

    const editor = window.activeTextEditor;
    if (editor) {
        const marker = mode === 'command' ? '?' : PROMPT_MARKER;
        isCommandMode = mode === 'command';
        isPromptMode = mode === 'prompt';
        await editor.edit(editBuilder => {
            editBuilder.insert(editor.selection.active, marker);
        });
        mode === 'command' ? highlightCommandMarker(editor) : highlightPromptMarker(editor);
    } else {
        console.log(`Promptly: Failed to enter ${mode} mode - no active editor`);
    }
}

async function handleTypeCommand(args: { text: string }) {
    if (args.text === '\n') {
        const editor = window.activeTextEditor;
        if (editor) {
            return handlePromptEnter(editor);
        }
    }
    return vscode.commands.executeCommand('default:type', args);
}


async function switchModel() {
    const config = workspace.getConfiguration('promptly');
    const availableModels = getAvailableModels();
    const setupLocalModelOption = 'Setup Local Model';
    const modelOptions = [...availableModels.filter(model => model !== setupLocalModelOption), setupLocalModelOption];

    const selectedModel = await window.showQuickPick(modelOptions, {
        placeHolder: 'Select a model',
    });

    if (selectedModel) {
        if (selectedModel === setupLocalModelOption) {
            await commands.executeCommand('promptly.setupLocalModel');
        } else {
            await config.update('model', selectedModel, ConfigurationTarget.Global);
            window.showInformationMessage(`Switched to model: ${selectedModel}.`);
            
            if (!selectedModel.startsWith('local:')) {
                await promptForApiKey(selectedModel, config);
            }
        }
    }
}

async function promptForApiKey(model: string, config: vscode.WorkspaceConfiguration) {
    const apiKeyConfig = model.startsWith('gemini-') ? 'geminiApiKey' :
                         model.startsWith('gpt-') ? 'openaiApiKey' :
                         model.startsWith('claude-') ? 'anthropicApiKey' : null;

    if (!apiKeyConfig) {
        throw new Error(`Unsupported model: ${model}`);
    }

    if (!config.get(apiKeyConfig)) {
        const setApiKey = await window.showInformationMessage(
            `API key for ${model} is not set. Do you want to set it now?`,
            'Yes', 'No'
        );
        if (setApiKey === 'Yes') {
            const apiKey = await window.showInputBox({
                prompt: `Enter your API key for ${model}`,
                password: true
            });
            if (apiKey) {
                await config.update(apiKeyConfig, apiKey, ConfigurationTarget.Global);
                window.showInformationMessage('API key has been set.');
            }
        }
    }
}