import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
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

let activePromptStatusBarItem: vscode.StatusBarItem;

export async function activate(context: ExtensionContext) {
    try {
        await updateModelList();
        registerCommands(context);
        startup();

        activePromptStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        activePromptStatusBarItem.command = 'promptly.selectActivePrompt';
        context.subscriptions.push(activePromptStatusBarItem);

        registerActivePromptCommand(context);

        updateActivePromptMenu();

        context.subscriptions.push(
            workspace.onDidChangeConfiguration(event => {
                if (event.affectsConfiguration('promptly.customPrompts') || 
                    event.affectsConfiguration('promptly.activePrompt')) {
                    updateActivePromptMenu();
                }

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

function registerActivePromptCommand(context: vscode.ExtensionContext) {
    const command = vscode.commands.registerCommand('promptly.selectActivePrompt', async () => {
        const config = vscode.workspace.getConfiguration('promptly');
        const customPrompts = config.get('customPrompts') as { [key: string]: string };

        const options = Object.keys(customPrompts).map(key => ({
            label: key,
            description: customPrompts[key].split('.')[0]
        }));

        const selected = await vscode.window.showQuickPick(options, {
            placeHolder: 'Select active prompt'
        });

        if (selected) {
            await config.update('activePrompt', selected.label, vscode.ConfigurationTarget.Global);
            vscode.window.showInformationMessage(`Active prompt set to: ${selected.label}`);
            updateActivePromptMenu();
        }
    });

    context.subscriptions.push(command);
}

function updateActivePromptMenu() {
    const config = vscode.workspace.getConfiguration('promptly');
    const customPrompts = config.get('customPrompts') as { [key: string]: string };
    const currentActivePrompt = config.get('activePrompt') as string;

    // Update status bar item
    if (activePromptStatusBarItem) {
        activePromptStatusBarItem.text = `Prompt: ${currentActivePrompt}`;
        activePromptStatusBarItem.show();
    }

    // Update the command palette
    commands.executeCommand('setContext', 'promptly:customPrompts', Object.keys(customPrompts));
}

function registerCommands(context: ExtensionContext) {
    const commandHandlers = {
        sendMessage: handleChat,
        extractCode: extractCodeFromLastResponse,
        sendMessageNotebook: handleChatNotebook,
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
    const config = vscode.workspace.getConfiguration('promptly');
    const hotkeys = config.get('hotkeys') as { [key: string]: string };

    // Update VS Code settings
    await config.update('hotkeys', hotkeys, vscode.ConfigurationTarget.Global);

    // Determine the path for keybindings.json based on VS Code version
    const isInsiders = vscode.env.appName.includes('Insiders');
    const vscodeUserPath = process.env.APPDATA ? 
        path.join(process.env.APPDATA, isInsiders ? 'Code - Insiders' : 'Code', 'User') :
        path.join(os.homedir(), '.config', isInsiders ? 'code-insiders' : 'code', 'User');
    const keybindingsPath = path.join(vscodeUserPath, 'keybindings.json');

    console.log(`Promptly: Using keybindings path: ${keybindingsPath}`);

    let keybindings: any[] = [];

    // Create the directory if it doesn't exist
    try {
        await fs.promises.mkdir(vscodeUserPath, { recursive: true });
    } catch (error) {
        console.error('Promptly: Error creating directory:', error);
        vscode.window.showErrorMessage('Error creating directory for keybindings.json');
        return;
    }

    // Read existing keybindings or create an empty file
    try {
        if (fs.existsSync(keybindingsPath)) {
            const keybindingsContent = await fs.promises.readFile(keybindingsPath, 'utf8');
            // Remove comments before parsing
            const jsonContent = keybindingsContent.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
            keybindings = JSON.parse(jsonContent);
        } else {
            // Create an empty keybindings.json file
            await fs.promises.writeFile(keybindingsPath, '[]', 'utf8');
            console.log('Promptly: Created new keybindings.json file');
        }
    } catch (error) {
        console.error('Promptly: Error reading/creating keybindings.json:', error);
        vscode.window.showErrorMessage('Error reading or creating keybindings.json. Using an empty array.');
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
        await fs.promises.writeFile(keybindingsPath, JSON.stringify(keybindings, null, 2), 'utf8');
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