// src/extension.ts
import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// Import from new structure
import { updateModelListSetting, getAvailableModels } from './config/modelService';
import { getActivePrompt } from './config/promptService';
import { handleChat } from './chat/chatService';
import { initializeDecorations, disposeDecorations, handleTextDocumentChange, handlePromptEnter } from './editor/markerHandler';
import { handleChatNotebook, handleTracebackError, handleNotebookDocumentChange } from './notebook/notebookService';
import { extractCodeCommand } from './features/codeExtraction'; // Use the combined command
import { startLocalModelSetup } from './server/localModelSetup'; // Keep setup command separate
import { handleServerCommand, resetLocalModelPort, setLocalModelPort /* potentially others */ } from './server/serverService';
import { UiChatViewProvider } from './ui/uiChat';
import { SETUP_LOCAL_MODEL_OPTION } from './common/constants'; // Import constant

import { OPENAI_GPT_REGEX } from './common/constants';
// State (if needed at extension level, otherwise keep in modules)
// let isPromptMode = false; // State likely managed within markerHandler now
// let isCommandMode = false;

let activePromptStatusBarItem: vscode.StatusBarItem;

export async function activate(context: vscode.ExtensionContext) {
    console.log('Activating Promptly Extension...');

    try {
        // --- Initialization ---
        initializeDecorations(); // Initialize editor decorations
        await updateModelListSetting(); // Check and update model setting on activation

        // --- UI Setup ---
        // Status Bar
        activePromptStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        activePromptStatusBarItem.command = 'promptly.selectActivePrompt';
        context.subscriptions.push(activePromptStatusBarItem);
        updateActivePromptMenu(); // Initial update

        // Webview Chat UI
        const uiChatViewProvider = new UiChatViewProvider(context.extensionUri);
        context.subscriptions.push(
            vscode.window.registerWebviewViewProvider('promptly-chat-view', uiChatViewProvider, {
                 webviewOptions: { retainContextWhenHidden: true } // Persist webview state
             })
        );


        // --- Command Registration ---
        registerCommands(context);

        // --- Event Listeners ---
        // Text Document Changes (for markers)
        context.subscriptions.push(
            vscode.workspace.onDidChangeTextDocument(handleTextDocumentChange)
        );
        // Notebook Document Changes (for errors)
        context.subscriptions.push(
            vscode.workspace.onDidChangeNotebookDocument(handleNotebookDocumentChange)
        );
         // Configuration Changes
         context.subscriptions.push(
             vscode.workspace.onDidChangeConfiguration(event => {
                 if (event.affectsConfiguration('promptly.customPrompts') ||
                     event.affectsConfiguration('promptly.activePrompt')) {
                     updateActivePromptMenu(); // Update status bar on prompt change
                 }
                 if (event.affectsConfiguration('promptly.model') ||
                      event.affectsConfiguration('promptly.localModelPath') || // React to local model changes
                      event.affectsConfiguration('promptly.localPreconfiguredModel')) {
                     updateModelListSetting(); // Re-validate selected model
                     // Potentially update UI or notify user
                 }
                 if (event.affectsConfiguration('promptly.hotkeys')) {
                     promptKeybindingUpdate();
                 }
                 // Add more config checks as needed
             })
         );
         // Handle 'type' command for Enter key interception in editor
         context.subscriptions.push(
             vscode.commands.registerCommand('type', handleTypeCommand)
         );


        console.log('Promptly Extension Activated Successfully.');

    } catch (error) {
        console.error('Promptly: Error during activation', error);
        vscode.window.showErrorMessage(`Promptly failed to activate: ${error}`);
         // Ensure cleanup if activation fails partially
         deactivate();
    }
}


function registerCommands(context: vscode.ExtensionContext) {
    // Core Chat Commands
     context.subscriptions.push(vscode.commands.registerCommand('promptly.sendMessage', () => handleChat()));
     context.subscriptions.push(vscode.commands.registerCommand('promptly.sendMessageNotebook', handleChatNotebook)); // Primarily for notebook context menu?

    // Feature Commands
     context.subscriptions.push(vscode.commands.registerCommand('promptly.extractCode', extractCodeCommand));
     context.subscriptions.push(vscode.commands.registerCommand('promptly.handleTracebackError', handleTracebackError)); // Command palette access for errors

    // Configuration/Setup Commands
     context.subscriptions.push(vscode.commands.registerCommand('promptly.switchModel', switchModelCommand));
     context.subscriptions.push(vscode.commands.registerCommand('promptly.setupLocalModel', startLocalModelSetup));
     context.subscriptions.push(vscode.commands.registerCommand('promptly.selectActivePrompt', selectActivePromptCommand));
     context.subscriptions.push(vscode.commands.registerCommand('promptly.updateKeybindings', updateKeybindingsCommand)); // Keep keybinding updater

     // UI Commands
      context.subscriptions.push(vscode.commands.registerCommand('promptly.openUiChat', () => {
         vscode.commands.executeCommand('workbench.view.extension.promptly-chat');
     }));

     // Potentially add commands for server status/shutdown from palette if needed
     // context.subscriptions.push(vscode.commands.registerCommand('promptly.serverStatus', () => handleServerCommand('status')));
     // context.subscriptions.push(vscode.commands.registerCommand('promptly.serverShutdown', () => handleServerCommand('shutdown')));
}

// --- Command Implementations ---

async function selectActivePromptCommand() {
     const config = vscode.workspace.getConfiguration('promptly');
     const customPrompts = config.get('customPrompts') as { [key: string]: string } | undefined;

     if (!customPrompts || Object.keys(customPrompts).length === 0) {
         vscode.window.showInformationMessage('No custom prompts configured in settings.');
         return;
     }

     const options = Object.keys(customPrompts).map(key => ({
         label: key,
         description: customPrompts[key].split('\n')[0].substring(0, 70) + '...' // Show first line as description
     }));

     const selected = await vscode.window.showQuickPick(options, {
         placeHolder: 'Select the active system prompt'
     });

     if (selected) {
         await config.update('activePrompt', selected.label, vscode.ConfigurationTarget.Global);
         // updateActivePromptMenu() will be called by the configuration listener
         vscode.window.showInformationMessage(`Active prompt set to: ${selected.label}`);
     }
 }

async function switchModelCommand() {
    const config = vscode.workspace.getConfiguration('promptly');
    const available = getAvailableModels(); // Get current list
    const options = [
        ...available,
        SETUP_LOCAL_MODEL_OPTION // Add setup option distinctly
    ];

    const selected = await vscode.window.showQuickPick(options, {
        placeHolder: 'Select LLM or Setup Local Model',
    });

    if (selected) {
        if (selected === SETUP_LOCAL_MODEL_OPTION) {
            await vscode.commands.executeCommand('promptly.setupLocalModel');
        } else {
            await config.update('model', selected, vscode.ConfigurationTarget.Global);
            vscode.window.showInformationMessage(`Switched to model: ${selected}`);
            // Prompt for API key if needed (moved logic here)
            if (!selected.startsWith('local:')) {
                 await promptForApiKeyIfMissing(selected, config);
             }
             // updateModelListSetting() will handle validation via config listener
        }
    }
}

// Helper to prompt for API key (similar to original, but called from switchModelCommand)
 async function promptForApiKeyIfMissing(model: string, config: vscode.WorkspaceConfiguration) {
    //  const { OPENAI_GPT_REGEX } = await import('./common/constants'); // Dynamic import if needed, or static
     let apiKeyConfig: string | null = null;
     let providerName: string | null = null;

     if (model.startsWith('gemini-')) { apiKeyConfig = 'geminiApiKey'; providerName = 'Gemini'; }
     else if (model.startsWith('claude-')) { apiKeyConfig = 'anthropicApiKey'; providerName = 'Anthropic'; }
     else if (model.startsWith('gpt-') || OPENAI_GPT_REGEX.test(model)) { apiKeyConfig = 'openaiApiKey'; providerName = 'OpenAI'; }

     if (apiKeyConfig && providerName && !config.get(apiKeyConfig)) {
         const setApiKey = await vscode.window.showInformationMessage(
             `API key for ${providerName} (${model}) is not set. Set it now?`,
             'Yes', 'No'
         );
         if (setApiKey === 'Yes') {
             const apiKey = await vscode.window.showInputBox({
                 prompt: `Enter your ${providerName} API key`,
                 password: true,
                 ignoreFocusOut: true
             });
             if (apiKey) {
                 await config.update(apiKeyConfig, apiKey, vscode.ConfigurationTarget.Global);
                 vscode.window.showInformationMessage(`${providerName} API key saved.`);
             }
         }
     }
 }


// --- UI Updates ---

function updateActivePromptMenu() {
    const config = vscode.workspace.getConfiguration('promptly');
    const activePromptName = config.get('activePrompt') as string | undefined;
    // const customPrompts = config.get('customPrompts') as { [key: string]: string } | undefined;

    if (activePromptStatusBarItem) {
        activePromptStatusBarItem.text = `$(hubot) Prompt: ${activePromptName || 'Default'}`; // Use an icon
        activePromptStatusBarItem.tooltip = `Active System Prompt: ${getActivePrompt().substring(0, 100)}... (Click to change)`;
        activePromptStatusBarItem.show();
    }

    // Update context for command palette visibility if needed (e.g., conditional commands)
    // vscode.commands.executeCommand('setContext', 'promptly:hasCustomPrompts', !!customPrompts && Object.keys(customPrompts).length > 0);
}

// --- Keybinding Update Logic (Keep as is for now) ---
async function updateKeybindingsCommand() {
    // This function remains complex due to direct file manipulation.
    // Consider using VS Code's configuration mechanisms more directly if possible in the future.
     const config = vscode.workspace.getConfiguration('promptly');
     const hotkeys = config.get('hotkeys') as { [key: string]: string } | undefined;

     if (!hotkeys) {
         vscode.window.showWarningMessage("No hotkeys defined in 'promptly.hotkeys' setting.");
         return;
     }

     // Determine the path for keybindings.json
     // This logic seems generally correct but relies on environment assumptions.
     const appName = vscode.env.appName;
     const isInsiders = appName.includes('Insiders');
     const isOSS = appName.includes('OSS') || appName.includes('Code - OSS'); // Handle OSS builds
     const appDir = isInsiders ? (isOSS ? 'Code - Insiders - OSS' : 'Code - Insiders') : (isOSS ? 'Code - OSS' : 'Code');

     let configPath: string;
     if (process.platform === 'win32') {
          configPath = path.join(process.env.APPDATA || '', appDir, 'User');
      } else if (process.platform === 'darwin') {
          configPath = path.join(os.homedir(), 'Library', 'Application Support', appDir, 'User');
      } else { // Linux
          configPath = path.join(os.homedir(), '.config', appDir, 'User');
      }

     const keybindingsPath = path.join(configPath, 'keybindings.json');
     console.log(`Promptly: Updating keybindings at: ${keybindingsPath}`);

     let keybindings: any[] = [];

     try {
         // Ensure directory exists
         await fs.promises.mkdir(configPath, { recursive: true });

         // Read existing keybindings or create file
         if (fs.existsSync(keybindingsPath)) {
             const keybindingsContent = await fs.promises.readFile(keybindingsPath, 'utf8');
             // Basic comment removal before parsing (can be fragile)
             const jsonContent = keybindingsContent.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
             try {
                  keybindings = JSON.parse(jsonContent || '[]'); // Handle empty or whitespace-only file
              } catch (parseError: any) {
                  console.error('Promptly: Error parsing keybindings.json:', parseError);
                  vscode.window.showErrorMessage(`Error parsing keybindings.json: ${parseError.message}. Please check the file for syntax errors.`);
                  return; // Stop if parsing fails
              }
         } else {
             await fs.promises.writeFile(keybindingsPath, '[]', 'utf8');
             console.log('Promptly: Created new keybindings.json file');
         }
     } catch (error: any) {
         console.error('Promptly: Error reading/creating keybindings.json:', error);
         vscode.window.showErrorMessage(`Error accessing keybindings.json: ${error.message}`);
         return; // Stop if file access fails
     }

     // Filter out existing Promptly keybindings
     keybindings = keybindings.filter(kb => !(kb.command && typeof kb.command === 'string' && kb.command.startsWith('promptly.')));

     // Add new Promptly keybindings from settings
     for (const [commandName, key] of Object.entries(hotkeys)) {
         if (key && typeof key === 'string') { // Ensure key is a non-empty string
             keybindings.push({
                 key: key,
                 command: `promptly.${commandName}`,
                 when: 'editorTextFocus' // Default context, maybe allow customization?
             });
         }
     }

     // Write updated keybindings back to file
     try {
         await fs.promises.writeFile(keybindingsPath, JSON.stringify(keybindings, null, 4), 'utf8'); // Use 4 spaces for indent
         vscode.window.showInformationMessage('Promptly: Keybindings updated successfully. You may need to restart VS Code for changes to take full effect.');
     } catch (error: any) {
         console.error('Promptly: Error writing keybindings.json:', error);
         vscode.window.showErrorMessage(`Error saving updated keybindings.json: ${error.message}`);
     }
 }

function promptKeybindingUpdate() {
     vscode.window.showInformationMessage(
         'Promptly: Hotkey configuration changed.',
         'Update Keybindings Now'
     ).then(selection => {
         if (selection === 'Update Keybindings Now') {
             vscode.commands.executeCommand('promptly.updateKeybindings');
         }
     });
 }

// --- Event Handlers / Interceptors ---

// Handle 'type' command to intercept Enter key
async function handleTypeCommand(args: { text: string }) {
    const editor = vscode.window.activeTextEditor;
    // Check if Enter key was pressed and an editor is active
    if (args.text === '\n' && editor) {
        // Delegate to markerHandler to check for prompt/command lines
         const result = await handlePromptEnter(editor);
         if (result === null) {
             return; // Suppress default Enter behavior as it was handled
         }
         // Otherwise, fall through to default behavior
    }

    // Execute default 'type' command if not handled
    return vscode.commands.executeCommand('default:type', args);
}


// --- Deactivation ---

export function deactivate(): Promise<void> {
    console.log('Deactivating Promptly Extension...');
    // Clean up resources
    disposeDecorations(); // Dispose editor decorations
    activePromptStatusBarItem?.dispose();
    // Dispose commands and listeners (handled by context.subscriptions)

     // Attempt to reset port state if server might have been running
     resetLocalModelPort(); // Reset port on deactivation

     // Cleanup renderer state if applicable
     // MessageRenderer.getInstance().dispose(); // If MessageRenderer is used and needs disposal

    console.log('Promptly Extension Deactivated.');
    return Promise.resolve();
}
