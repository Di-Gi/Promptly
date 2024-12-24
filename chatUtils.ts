// chatUtils.ts
import * as vscode from 'vscode';
import { window, workspace, Range, TextEditor, Position, ConfigurationTarget, NotebookEditor} from 'vscode';
import { getServerStats, shutdownServer, getLocalModelPort } from './localModelSetup';

const RESPONSE_START_MARKER = '\u200B⚡RESPONSE_START⚡\u200B';
const RESPONSE_END_MARKER = '\u200B●RESPONSE_END●\u200B';
const PROMPT_MARKER = '>>';

const GREEN_COLOR = new vscode.ThemeColor('editorInfo.foreground');
const BLUE_COLOR = new vscode.ThemeColor('editorInfo.foreground');
const WHITE_COLOR = new vscode.ThemeColor('editor.foreground');

let lastResponseRange: Range | null = null;
let storedCodeBlocks: { code: string; language: string | null }[] = [];

let isPromptMode = false;
let isCommandMode = false;
let promptDecoration: vscode.TextEditorDecorationType;
let commandDecoration: vscode.TextEditorDecorationType;

let isProcessingRequest = false;

// Dictionary of comment characters for different file types
const commentChars: { [key: string]: string } = {
    'typescript': '//',
    'javascript': '//',
    'python': '#',
    'ruby': '#',
    'java': '//',
    'c': '//',
    'cpp': '//',
    'csharp': '//',
    'go': '//',
    'rust': '//',
    'swift': '//',
    'php': '//',
    'html': '<!--',
    'css': '/*',
    'scss': '//',
    'less': '//',
    'xml': '<!--',
    'markdown': '<!--',
    'yaml': '#',
    'json': '//',
    'plaintext': '//',
  };

interface LocalModelResponse {
    generated_texts: string[];
}

interface GeminiResponse {
    candidates?: Array<{
        content?: {
            parts?: Array<{
                text?: string;
            }>;
        };
    }>;
}

interface ResponsePart {
    type: 'code' | 'text';
    content: string;
    language?: string;
}

interface AnimationControl {
    stop: () => Promise<void>;
    animationRange: vscode.Range;
    insertPosition: vscode.Position;
}

interface OpenAIResponse {
    choices: Array<{
        message: {
            content: string;
        };
    }>;
}

interface AnthropicResponse {
    content: Array<{
        type: string;
        text: string;
    }>;
}

export function startup() {
    promptDecoration = window.createTextEditorDecorationType({ color: GREEN_COLOR });
    commandDecoration = window.createTextEditorDecorationType({ color: BLUE_COLOR });

}

function getAvailableModels(): string[] {
    const extensionId = 'digi.promptly';
    const extension = vscode.extensions.getExtension(extensionId);
    let models: string[] = [];

    try {
        if (extension) {
            const packageJSON = extension.packageJSON;
            const modelConfig = packageJSON.contributes?.configuration?.properties?.['promptly.model'];
            
            if (modelConfig?.oneOf && Array.isArray(modelConfig.oneOf)) {
                // Define a type for the items in modelConfig.oneOf
                type ModelConfigItem = {
                    enum?: string[];
                    pattern?: string;
                };

                const modelEnum = modelConfig.oneOf.find((item: ModelConfigItem) => Array.isArray(item.enum))?.enum;
                if (Array.isArray(modelEnum)) {
                    models = modelEnum.filter(model => model !== 'Setup Local Model');
                }
            }
        }
    } catch (error) {
        console.error('Error reading model list from package.json:', error);
    }

    if (models.length === 0) {
        console.warn('Unable to retrieve model list from package.json. Using fallback list.');
        models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro-exp-0801",
            "gemini-1.5-pro",
            "gpt-3.5-turbo",
            "gpt-4",
            "claude-2",
            "claude-instant-1"
        ];
    }

    // Add available local models
    const localModels = getLocalModels();
    models = [...models, ...localModels];

    console.log('Available models:', models);
    return models;
}

function getLocalModels(): string[] {
    const config = vscode.workspace.getConfiguration('promptly');
    const localModelPath = config.get('localModelPath') as string;
    const localPreconfiguredModel = config.get('localPreconfiguredModel') as string;
    if (localModelPath && localPreconfiguredModel) {
        return [`local:${localPreconfiguredModel}`];
    }
    return [];
}

async function updateModelList() {
    const config = workspace.getConfiguration('promptly');
    const allModels = getAvailableModels();
    const currentModel = config.get('model') as string;
    if (!allModels.includes(currentModel)) {
        // If the current model is no longer available, switch to the first available model
        await config.update('model', allModels[0], ConfigurationTarget.Global);
        window.showInformationMessage(`The previously selected model is no longer available. Switched to: ${allModels[0]}`);
    }
}

function doesPromptMarkerExist(editor: vscode.TextEditor): boolean {
    const document = editor.document;
    for (let i = 0; i < document.lineCount; i++) {
        const lineText = document.lineAt(i).text;
        // Check if the line contains the prompt marker and it's not part of a comment closing
        if (lineText.includes(PROMPT_MARKER) && !lineText.trim().endsWith('-->')) {
            return true;
        }
    }
    return false;
}

export async function handleCommand(command: string): Promise<void> {
    const [action, ...args] = command.split(' ');
    switch (action.toLowerCase()) {
        case 'status':
            await handleStatusCommand(args);
            break;
        case 'shutdown':
            await handleShutdownCommand(args);
            break;
        default:
            vscode.window.showInformationMessage(`Unknown command: ${action}`);
    }
}

async function handleStatusCommand(args: string[]): Promise<void> {
    if (args[0]?.toLowerCase() === 'server') {
        const stats = await getServerStats();
        if (stats) {
            vscode.window.showInformationMessage(`Server Stats: CPU: ${stats.cpu_percent.toFixed(2)}%, ` +
                `Memory: ${stats.memory_percent.toFixed(2)}%, ` +
                `Queue Size: ${stats.queue_size}`);
        }
        else {
            vscode.window.showErrorMessage('Failed to fetch server stats. Make sure the model server is running.');
        }
    }
    else {
        vscode.window.showInformationMessage('Unknown status command. Available: status server');
    }
}

async function handleShutdownCommand(args: string[]): Promise<void> {
    if (args[0]?.toLowerCase() === 'server') {
        const confirmation = await vscode.window.showWarningMessage('Are you sure you want to shut down the server?', 'Yes', 'No');
        if (confirmation === 'Yes') {
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Shutting down server",
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: "Initiating shutdown..." });
                try {
                    const success = await shutdownServer();
                    if (success) {
                        progress.report({ increment: 100, message: "Server shutdown complete" });
                        vscode.window.showInformationMessage('Server has been shut down successfully.');
                    }
                    else {
                        throw new Error('Shutdown request failed');
                    }
                }
                catch (error: unknown) {
                    console.error('Error during server shutdown:', error);
                    progress.report({ increment: 100, message: "Shutdown failed" });
                    if (error instanceof Error) {
                        vscode.window.showErrorMessage(`Failed to shut down the server: ${error.message}`);
                    }
                    else {
                        vscode.window.showErrorMessage(`Failed to shut down the server: ${String(error)}`);
                    }
                }
            });
        }
    }
    else {
        vscode.window.showInformationMessage('Unknown shutdown command. Available: shutdown server');
    }
}

async function startRequestAnimation(editor: vscode.TextEditor, insertPosition: vscode.Position): Promise<AnimationControl> {
    let dotCount = 0;
    const animationStartPosition = new vscode.Position(insertPosition.line + 2, 0);
    let interval: NodeJS.Timeout;
    const blueDotDecoration = vscode.window.createTextEditorDecorationType({
        color: new vscode.ThemeColor('editorInfo.foreground')
    });
    const stopAnimation = async () => {
        clearInterval(interval);
        blueDotDecoration.dispose();
        // Remove the animation dots
        await editor.edit(editBuilder => {
            const rangeToDelete = new vscode.Range(animationStartPosition, new vscode.Position(animationStartPosition.line, animationStartPosition.character + 3));
            editBuilder.delete(rangeToDelete);
        });
    };
    await editor.edit(editBuilder => {
        if (insertPosition.character !== 0) {
            editBuilder.insert(insertPosition, '\n\n');
        }
        else {
            editBuilder.insert(insertPosition, '\n');
        }
    });
    interval = setInterval(() => {
        editor.edit(editBuilder => {
            if (dotCount > 0) {
                const startPos = animationStartPosition;
                const endPos = new vscode.Position(startPos.line, startPos.character + dotCount);
                editBuilder.delete(new vscode.Range(startPos, endPos));
            }
            dotCount = (dotCount % 3) + 1;
            editBuilder.insert(animationStartPosition, '.'.repeat(dotCount));
        }).then(() => {
            const dotRange = new vscode.Range(animationStartPosition, new vscode.Position(animationStartPosition.line, animationStartPosition.character + dotCount));
            editor.setDecorations(blueDotDecoration, [dotRange]);
        });
    }, 500);
    return {
        stop: stopAnimation,
        animationRange: new vscode.Range(animationStartPosition, new vscode.Position(animationStartPosition.line, animationStartPosition.character + 3)),
        insertPosition: animationStartPosition
    };
}

export async function handleTracebackError() {
    const notebookEditor = vscode.window.activeNotebookEditor;
    if (notebookEditor) {
        const notebook = notebookEditor.notebook;
        const lastExecutedCell = findLastExecutedCell(notebook);
        if (lastExecutedCell) {
            const error = detectTracebackError(lastExecutedCell);
            if (error) {
                await handleChatNotebook(notebook, lastExecutedCell, error);
            } else {
                vscode.window.showInformationMessage('No traceback error detected in the last executed cell of the current notebook.');
            }
        } else {
            vscode.window.showInformationMessage('No executed cells found in the current notebook.');
        }
    } else {
        vscode.window.showInformationMessage('No active Jupyter Notebook found.');
    }
}


function detectTracebackError(cell: vscode.NotebookCell): string | null {
    if (cell.kind !== vscode.NotebookCellKind.Code || cell.outputs.length === 0) {
        return null;
    }
    for (const output of cell.outputs) {
        for (const item of output.items) {
            if (item.mime === 'application/vnd.code.notebook.error') {
                const errorData = JSON.parse(Buffer.from(item.data).toString('utf8'));
                const tracebackText = `${errorData.name}: ${errorData.message}\n${errorData.stack}`;
                return tracebackText;
            }
            else if (item.mime === 'application/vnd.code.notebook.stdout' ||
                item.mime === 'application/vnd.code.notebook.stderr' ||
                item.mime === 'text/plain') {
                const outputText = Buffer.from(item.data).toString('utf8');
                if (outputText.includes('Traceback') ||
                    outputText.includes('Error:') ||
                    outputText.includes('Exception') ||
                    outputText.toLowerCase().includes('error')) {
                    return outputText;
                }
            }
        }
    }
    console.log('No traceback error detected in the cell');
    return null;
}

async function handleNotebookDocumentChange(e: vscode.NotebookDocumentChangeEvent) {
    // Check if the change involves cell execution
    const hasExecutionChanges = e.cellChanges.some(change => change.executionSummary !== undefined);
    if (hasExecutionChanges) {
        const notebook = e.notebook;
        const lastExecutedCell = findLastExecutedCell(notebook);
        if (lastExecutedCell) {
            const error = detectTracebackError(lastExecutedCell);
            if (error) {
                // Directly handle the error without prompting the user
                await handleChatNotebook(notebook, lastExecutedCell, error);
            }
            else {
                console.log('No error detected in the last executed cell');
            }
        }
        else {
            console.log('No last executed cell found');
        }
    }
    else {
        console.log('No execution changes detected');
    }
}

function findLastExecutedCell(notebook: vscode.NotebookDocument): vscode.NotebookCell | undefined {
    const cells = notebook.getCells().reverse();
    for (const cell of cells) {
        if (cell.kind === vscode.NotebookCellKind.Code &&
            cell.outputs.length > 0 &&
            cell.executionSummary?.executionOrder !== undefined) {
            console.log('Found last executed cell:', cell.index, 'Execution order:', cell.executionSummary.executionOrder);
            return cell;
        }
    }
    console.log('No executed cells found');
    return undefined;
}

async function startNotebookLoadingAnimation(notebookEditor: vscode.NotebookEditor): Promise<() => Promise<void>> {
    const notebook = notebookEditor.notebook;
    const insertIndex = notebook.cellCount;
    const loadingCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Markup, 'Loading response...', 'markdown');
    const edit = new vscode.WorkspaceEdit();
    const range = new vscode.NotebookRange(insertIndex, insertIndex);
    edit.set(notebook.uri, [vscode.NotebookEdit.insertCells(range.start, [loadingCell])]);
    await vscode.workspace.applyEdit(edit);
    let dots = 0;
    const interval = setInterval(() => {
        dots = (dots + 1) % 4;
        const loadingText = 'Loading response' + '.'.repeat(dots);
        const cellEdit = new vscode.WorkspaceEdit();
        cellEdit.set(notebook.uri, [vscode.NotebookEdit.updateCellMetadata(insertIndex, { custom: { loading: loadingText } })]);
        vscode.workspace.applyEdit(cellEdit);
    }, 500);
    return async () => {
        clearInterval(interval);
        const removeEdit = new vscode.WorkspaceEdit();
        removeEdit.set(notebook.uri, [vscode.NotebookEdit.deleteCells(new vscode.NotebookRange(insertIndex, insertIndex + 1))]);
        await vscode.workspace.applyEdit(removeEdit);
    };
}

async function handleChatNotebook(notebook?: vscode.NotebookDocument, errorCell?: vscode.NotebookCell, errorText?: string) {
    const notebookEditor = vscode.window.activeNotebookEditor;
    if (!notebookEditor) {
        vscode.window.showInformationMessage('No active Jupyter Notebook found.');
        return;
    }
    if (!notebook) {
        notebook = notebookEditor.notebook;
    }
    
    const config = vscode.workspace.getConfiguration('promptly');
    const currentModel = getCurrentModel();
    let apiKey: string | undefined;

    if (currentModel.startsWith('gemini-')) {
        apiKey = config.get('geminiApiKey') as string;
    } else if (currentModel.startsWith('gpt-')) {
        apiKey = config.get('openaiApiKey') as string;
    } else if (currentModel.startsWith('claude-')) {
        apiKey = config.get('anthropicApiKey') as string;
    } else if (currentModel.startsWith('local:')) {
        // No API key needed for local models
        apiKey = 'local';
    }

    if (!apiKey) {
        vscode.window.showErrorMessage(`API key not configured for ${currentModel}. Please set the appropriate API key in your settings.`);
        return;
    }

    let prompt: string;
    if (errorCell && errorText) {
        const userChoice = await vscode.window.showQuickPick(['Yes', 'No'], {
            placeHolder: 'Error detected. Do you want to ask for help with this error?'
        });
        if (userChoice !== 'Yes') {
            return; // Exit if user doesn't want help
        }
        prompt = `I encountered the following error in my Jupyter Notebook:
  
        Code:
        ${errorCell.document.getText()}
        
        Error:
        ${errorText}
        
        Can you explain what's causing this error and how to fix it?`;
    } else {
        console.log('No error detected, proceeding with normal cell selection');
        const normalSelection = await getNormalCellSelection(notebookEditor);
        if (!normalSelection) {
            return; // Exit if no valid selection was made
        }
        prompt = normalSelection;
    }

    // Start the loading animation
    const stopLoadingAnimation = await startNotebookLoadingAnimation(notebookEditor);
    try {
        const response = await sendMessage(prompt);
        console.log('Received response from API:', response);
        // Stop the loading animation
        await stopLoadingAnimation();
        await appendResponseToNotebook(notebookEditor, response);
    } catch (error: unknown) {
        console.error("Error in handleChatNotebook:", error);
        // Stop the loading animation
        await stopLoadingAnimation();
        if (error instanceof Error) {
            vscode.window.showErrorMessage(`Error: ${error.message}`);
        } else {
            vscode.window.showErrorMessage(`An unexpected error occurred`);
        }
    }
}

async function getNormalCellSelection(notebookEditor: vscode.NotebookEditor): Promise<string | undefined> {
    const selectedCells = notebookEditor.selections.flatMap(selection => notebookEditor.notebook.getCells(selection).filter((cell: vscode.NotebookCell) => cell.kind === vscode.NotebookCellKind.Code));
    if (selectedCells.length === 0) {
        vscode.window.showInformationMessage('No code cells selected. Please select at least one code cell.');
        return undefined;
    }
    return selectedCells.map(cell => cell.document.getText()).join('\n\n');
}

async function extractCodeFromNotebook(notebookEditor: NotebookEditor) {
    const notebook = notebookEditor.notebook;
    const cells = notebook.getCells();
    const lastMarkdownCell = cells.reverse().find(cell => cell.kind === vscode.NotebookCellKind.Markup);
    if (!lastMarkdownCell) {
        window.showInformationMessage('No markdown cell found in the notebook.');
        return;
    }
    const markdownContent = lastMarkdownCell.document.getText();
    const codeBlocks = markdownContent.match(/```[\s\S]*?```/g);
    if (!codeBlocks || codeBlocks.length === 0) {
        window.showInformationMessage('No code blocks found in the last markdown cell.');
        return;
    }
    storedCodeBlocks = codeBlocks.map(extractCodeAndLanguage);
    if (storedCodeBlocks.length <= 3) {
        for (const block of storedCodeBlocks) {
            await insertCodeBlockToNotebook(notebookEditor, block);
        }
        window.showInformationMessage('All code blocks have been inserted.');
        storedCodeBlocks = [];
    }
    else {
        await showCodeBlockPicker(notebookEditor);
    }
}

async function insertCodeBlockToNotebook(notebookEditor: NotebookEditor, codeBlock: {
    code: string;
    language: string | null;
}) {
    const notebook = notebookEditor.notebook;
    const insertIndex = notebook.cellCount;
    const codeCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, codeBlock.code, codeBlock.language || 'python');
    const edit = new vscode.WorkspaceEdit();
    const range = new vscode.NotebookRange(insertIndex, insertIndex);
    edit.set(notebook.uri, [vscode.NotebookEdit.insertCells(range.start, [codeCell])]);
    await vscode.workspace.applyEdit(edit);
    window.showInformationMessage(`Code block inserted. ${codeBlock.language ? `Language: ${codeBlock.language}` : 'No specific language detected.'}`);
}

async function appendResponseToNotebook(notebookEditor: vscode.NotebookEditor, response: string) {
    const notebook = notebookEditor.notebook;
    const insertIndex = notebook.cellCount;
    // Split the response into code blocks and text
    const parts = splitResponseIntoParts(response);
    const cellsToAdd: vscode.NotebookCellData[] = [];
    for (const part of parts) {
        if (part.type === 'code') {
            const codeCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, part.content, part.language || 'python');
            cellsToAdd.push(codeCell);
        }
        else {
            const markdownCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Markup, part.content, 'markdown');
            cellsToAdd.push(markdownCell);
        }
    }
    const edit = new vscode.WorkspaceEdit();
    const range = new vscode.NotebookRange(insertIndex, insertIndex);
    edit.set(notebook.uri, [vscode.NotebookEdit.insertCells(range.start, cellsToAdd)]);
    try {
        const success = await vscode.workspace.applyEdit(edit);
        if (success) {
            console.log('Response appended to Jupyter Notebook');
            vscode.window.showInformationMessage('Response added to notebook.');
        }
        else {
            console.error('Failed to append response to notebook');
            vscode.window.showErrorMessage('Failed to add response to notebook.');
        }
    }
    catch (error) {
        console.error('Error appending response to notebook:', error);
        vscode.window.showErrorMessage('Error adding response to notebook.');
    }
}

function splitResponseIntoParts(response: string): ResponsePart[] {
    const parts: ResponsePart[] = [];
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let match;
    while ((match = codeBlockRegex.exec(response)) !== null) {
        // Add text before the code block
        if (match.index > lastIndex) {
            parts.push({
                type: 'text',
                content: response.slice(lastIndex, match.index).trim()
            });
        }
        // Add the code block
        parts.push({
            type: 'code',
            content: match[2].trim(),
            language: match[1] || 'python' // Default to python if no language is specified
        });
        lastIndex = match.index + match[0].length;
    }
    // Add any remaining text after the last code block
    if (lastIndex < response.length) {
        parts.push({
            type: 'text',
            content: response.slice(lastIndex).trim()
        });
    }
    return parts;
}

async function handleChat(promptOverride?: string) {
    if (isProcessingRequest) {
        vscode.window.showInformationMessage('A request is already being processed. Please wait.');
        return;
    }

    isProcessingRequest = true;

    const editor = vscode.window.activeTextEditor;
    const notebookEditor = vscode.window.activeNotebookEditor;
    if (notebookEditor) {
        await handleChatNotebook();
        isProcessingRequest = false;
        return;
    }
    if (!editor) {
        vscode.window.showInformationMessage('No active editor found.');
        isProcessingRequest = false;
        return;
    }
    let prompt = promptOverride || await getPrompt(editor);
    let insertPosition = getInsertPosition(editor);
    const fileType = editor.document.languageId;
    let animationControl: AnimationControl | undefined;
    try {
        storedCodeBlocks = [];
        lastResponseRange = null;
        animationControl = await startRequestAnimation(editor, insertPosition);
        // Increase the timeout to 5 minutes (300000 ms)
        const timeoutPromise = new Promise<never>((_, reject) => {
            setTimeout(() => reject(new Error('Request timed out')), 300000);
        });
        const responsePromise = sendMessage(prompt);
        const response = await Promise.race([responsePromise, timeoutPromise]);
        if (animationControl) {
            await animationControl.stop();
        }
        if (response) {
            await appendResponseToFile(editor, response, animationControl, fileType);
        }
        else {
            vscode.window.showErrorMessage('No response received from the model.');
        }
    }
    catch (error: unknown) {
        console.error("Error in handleChat:", error);
        if (animationControl) {
            await animationControl.stop();
        }
        if (error instanceof Error) {
            vscode.window.showErrorMessage(`Error: ${error.message}`);
        }
        else {
            vscode.window.showErrorMessage(`An unexpected error occurred`);
        }
    }
    finally {
        isProcessingRequest = false;
    }
}

function getInsertPosition(editor: TextEditor): Position {
    const selection = editor.selection;
    if (!selection.isEmpty) {
        // If there's a selection, use the end of the selection
        return selection.end;
    }
    else {
        // If no selection, use the current cursor position
        return editor.selection.active;
    }
}

async function getPrompt(editor: TextEditor): Promise<string> {
    const selection = editor.selection;
    let prompt: string;
    if (!selection.isEmpty) {
        // If there's a selection, use it as the prompt
        prompt = editor.document.getText(selection);
    }
    else {
        // If no selection, use the entire document content
        prompt = editor.document.getText();
    }
    return prompt.trim();
}


function getCurrentModel(): string {
    const config = workspace.getConfiguration('promptly');
    const rawModel = config.get('model') as string;
    console.log('Raw model from configuration:', rawModel);
    const availableModels = getAvailableModels();
    console.log('Available models:', availableModels);
    // If the raw model is in the list of available models, use it
    if (availableModels.includes(rawModel)) {
        console.log('Returning model:', rawModel);
        return rawModel;
    }
    // If the model is not in the available list, fall back to the default
    console.warn('Model not found in available models. Falling back to default.');
    return availableModels[0] || 'gemini-1.5-flash'; // First available model or a hardcoded default
}

export function getActivePrompt(): string {
    const config = vscode.workspace.getConfiguration('promptly');
    const customPrompts = config.get('customPrompts') as { [key: string]: string };
    const activePromptName = config.get('activePrompt') as string;
    return customPrompts[activePromptName] || customPrompts['default'];
}

async function sendMessage(message: string): Promise<string> {
    const model = getCurrentModel();
    console.log('Model being used:', model);
    const config = workspace.getConfiguration('promptly');
    
    // Get the active prompt
    const systemPrompt = getActivePrompt();

    let apiKeyConfig: string;
    if (model.startsWith('gemini-')) {
        apiKeyConfig = 'geminiApiKey';
    } else if (model.startsWith('gpt-')) {
        apiKeyConfig = 'openaiApiKey';
    } else if (model.startsWith('claude-')) {
        apiKeyConfig = 'anthropicApiKey';
    } else if (model.startsWith('local:')) {
        return sendLocalModelMessage(message, systemPrompt);
    } else {
        throw new Error(`Unsupported model: ${model}`);
    }

    const apiKey = config.get(apiKeyConfig) as string;
    if (!apiKey) {
        throw new Error(`${apiKeyConfig} not configured. Please set it in your settings.`);
    }

    // Create a timeout promise
    const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('Request timed out')), 300000); // 5 minutes timeout
    });

    let responsePromise: Promise<string>;
    switch (true) {
        case model.startsWith('gemini-'):
            responsePromise = sendGeminiMessage(message, model, apiKey, systemPrompt);
            break;
        case model.startsWith('gpt-'):
            responsePromise = sendOpenAIMessage(message, model, apiKey, systemPrompt);
            break;
        case model.startsWith('claude-'):
            responsePromise = sendAnthropicMessage(message, model, apiKey, systemPrompt);
            break;
        default:
            throw new Error(`Unsupported model: ${model}`);
    }

    // Race the response promise against the timeout
    return Promise.race([responsePromise, timeoutPromise]);
}




async function sendLocalModelMessage(message: string, systemPrompt: string): Promise<string> {
    try {
        const port = await getLocalModelPort();
        const response = await fetch(`http://localhost:${port}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                prompt: `${systemPrompt}\n\nUser: ${message}\n\nAssistant:`,
                max_length: 512
            }),
            // Increased the timeout to 5 minutes (300000 milliseconds)
            signal: AbortSignal.timeout(300000)
        });

        if (!response.ok) {
            if (response.status === 422) {
                const errorData = await response.json();
                throw new Error(`Invalid request: ${JSON.stringify(errorData)}`);
            }
            throw new Error(`Local model API request failed with status ${response.status}`);
        }

        const data = await response.json() as LocalModelResponse;
        console.log('Local model response:', data);

        if (!data || !Array.isArray(data.generated_texts) || data.generated_texts.length === 0) {
            throw new Error('Invalid response format from local model');
        }

        return data.generated_texts[0];
    } catch (error) {
        console.error('Error sending message to local model:', error);
        if (error instanceof Error) {
            throw new Error(`Failed to communicate with the local model: ${error.message}`);
        } else {
            throw new Error('Failed to communicate with the local model. Make sure the model server is running.');
        }
    }
}

async function sendOpenAIMessage(message: string, model: string, apiKey: string, systemPrompt: string): Promise<string> {
    const url = 'https://api.openai.com/v1/chat/completions';
    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
    };
    const body = JSON.stringify({
        model: model,
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: message }
        ],
    });
    const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: body
    });
    if (!response.ok) {
        throw new Error(`OpenAI API request failed with status ${response.status}`);
    }
    const data = await response.json() as OpenAIResponse;
    if (!data.choices || !data.choices[0] || !data.choices[0].message || typeof data.choices[0].message.content !== 'string') {
        throw new Error('Unexpected response format from OpenAI API');
    }
    return data.choices[0].message.content;
}

async function sendAnthropicMessage(message: string, model: string, apiKey: string, systemPrompt: string): Promise<string> {
    const url = 'https://api.anthropic.com/v1/messages';
    const headers = {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
    };
    const body = JSON.stringify({
        model: model,
        max_tokens: 1024,
        messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: message }
        ]
    });
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: body
        });
        if (!response.ok) {
            throw new Error(`Anthropic API request failed with status ${response.status}`);
        }
        const data = await response.json() as AnthropicResponse;
        if (data.content[0]?.type !== 'text' || typeof data.content[0]?.text !== 'string') {
            throw new Error('Unexpected response format from Anthropic API');
        }
        return data.content[0].text;
    }
    catch (error) {
        if (error instanceof Error) {
            throw new Error(`Anthropic API request failed: ${error.message}`);
        }
        else {
            throw new Error('An unknown error occurred');
        }
    }
}

async function sendGeminiMessage(message: string, model: string, apiKey: string, systemPrompt: string): Promise<string> {
    const maxRetries = 3;
    const retryDelay = 1000; // 1 second
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${apiKey}`;
            const headers = {
                'Content-Type': 'application/json',
            };
            const body = JSON.stringify({
                contents: [
                    {
                        parts: [
                            { text: systemPrompt },
                            { text: message }
                        ]
                    }
                ],
                generationConfig: {
                    temperature: 0.78,
                    topP: 0.95,
                    topK: 64,
                    maxOutputTokens: 8192,
                },
            });
            console.log(`Sending request to Gemini API (Attempt ${attempt})...`);
            console.log("URL:", url.replace(apiKey, '...')); // Log URL with redacted API key
            console.log("Headers:", JSON.stringify(headers, null, 2));
            console.log("Request Body:", body);
            const response = await fetch(url, {
                method: 'POST',
                headers: headers,
                body: body,
            });
            console.log("Response status:", response.status);
            console.log("Response status text:", response.statusText);
            console.log("Response headers:", JSON.stringify(Object.fromEntries(response.headers), null, 2));
            if (!response.ok) {
                const errorBody = await response.text();
                console.error("Error response body:", errorBody);
                throw new Error(`Gemini API request failed with status ${response.status}: ${errorBody}`);
            }
            console.log("Gemini API response received successfully.");
            const responseBody = await response.json() as GeminiResponse;
            console.log("Full response:", JSON.stringify(responseBody, null, 2));
            if (responseBody.candidates && responseBody.candidates.length > 0) {
                const candidate = responseBody.candidates[0];
                if (candidate.content?.parts && candidate.content.parts.length > 0) {
                    const fullText = candidate.content.parts[0].text;
                    if (fullText) {
                        console.log("Extracted text:", fullText);
                        return fullText;
                    }
                }
            }
            throw new Error("No valid response content found in the API response");
        }
        catch (error) {
            console.error(`Error sending message to Gemini API (Attempt ${attempt}):`, error);
            if (attempt === maxRetries) {
                if (error instanceof Error) {
                    throw new Error(`Failed to send message to Gemini API after ${maxRetries} attempts: ${error.message}`);
                }
                else {
                    throw new Error(`Failed to send message to Gemini API after ${maxRetries} attempts: Unknown error`);
                }
            }
            await new Promise(resolve => setTimeout(resolve, retryDelay));
        }
    }
    // This line should never be reached due to the throw in the for loop, but TypeScript requires it
    throw new Error("Unexpected error in sendGeminiMessage");
}


// Message State Management
interface MessageSegment {
    id: string;
    text: string;
    lineNumber: number;
    position: vscode.Position;
    processed: boolean;
    written: boolean;
    writtenPosition?: vscode.Position;
    retryCount: number;
    checksum?: string;
}

interface EditorState {
    isActive: boolean;
    lastValidPosition?: vscode.Position;
    contentComplete: boolean;
    headerWritten: boolean;
    headerPosition?: vscode.Position;
    lastProcessedSegmentId?: string;
    segmentOrder: string[];
}

interface PendingMessage {
    id: string;
    message: string;
    segments: Map<string, MessageSegment>;
    position: vscode.Position;
    fileType: string;
    isComplete: boolean;
    retryCount: number;
    editorState: EditorState;
    editor: vscode.TextEditor;
    decorations?: {
        blueMarker: vscode.TextEditorDecorationType;
        whiteText: vscode.TextEditorDecorationType;
        startRange?: vscode.Range;
        endRange?: vscode.Range;
        contentRange?: vscode.Range;
    };
}

class MessageStateManager {
    private messages: Map<string, PendingMessage> = new Map();
    private activeDocumentId?: string;

    generateId(): string {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    createChecksum(content: string): string {
        return content.split('').reduce((acc, char) => 
            (((acc << 5) - acc) + char.charCodeAt(0))|0, 0).toString(36);
    }

    segmentMessage(message: string, startPosition: vscode.Position): [Map<string, MessageSegment>, string[]] {
        const segments = new Map<string, MessageSegment>();
        const segmentOrder: string[] = [];
        
        message.split('\n').forEach((line, index) => {
            const segmentId = this.generateId();
            segments.set(segmentId, {
                id: segmentId,
                text: line,
                lineNumber: startPosition.line + index + 3,
                position: new vscode.Position(startPosition.line + index + 3, 0),
                processed: false,
                written: false,
                retryCount: 0,
                checksum: this.createChecksum(line)
            });
            segmentOrder.push(segmentId);
        });

        return [segments, segmentOrder];
    }

    async validateContent(
        editor: vscode.TextEditor,
        pendingMessage: PendingMessage
    ): Promise<{ isValid: boolean; lastValidSegmentId?: string }> {
        const document = editor.document;
        let lastValidSegmentId: string | undefined;
        const startLine = pendingMessage.editorState.headerPosition?.line || pendingMessage.position.line;

        for (const segmentId of pendingMessage.editorState.segmentOrder) {
            const segment = pendingMessage.segments.get(segmentId);
            if (!segment?.written) {continue;}

            try {
                const lineContent = document.lineAt(startLine + segment.lineNumber).text;
                const currentChecksum = this.createChecksum(lineContent.trim());
                
                if (currentChecksum === segment.checksum) {
                    lastValidSegmentId = segmentId;
                } else {
                    break;
                }
            } catch (error) {
                break;
            }
        }

        return {
            isValid: lastValidSegmentId === pendingMessage.editorState.segmentOrder[
                pendingMessage.editorState.segmentOrder.length - 1
            ],
            lastValidSegmentId
        };
    }

    async validateEditorState(
        editor: vscode.TextEditor,
        pendingMessage: PendingMessage
    ): Promise<boolean> {
        try {
            if (!editor || !editor.document || editor.document.isClosed) {
                return false;
            }

            const activeEditor = vscode.window.activeTextEditor;
            if (!activeEditor || activeEditor.document.uri.toString() !== editor.document.uri.toString()) {
                return false;
            }

            try {
                await editor.edit(editBuilder => {
                    editBuilder.insert(new vscode.Position(0, 0), '');
                });
                return true;
            } catch {
                return false;
            }
        } catch (error) {
            console.error('Editor state validation failed:', error);
            return false;
        }
    }

    private async processSegment(
        editor: vscode.TextEditor,
        segment: MessageSegment,
        currentPosition: vscode.Position,
        needsClosingTag: boolean,
        commentChar: string,
        pendingMessage: PendingMessage
    ): Promise<vscode.Position | null> {
        if (!await this.validateEditorState(editor, pendingMessage)) {
            return null;
        }

        const chunkSize = 5;
        try {
            let segmentPosition = currentPosition;

            if (!needsClosingTag) {
                await editor.edit(editBuilder => {
                    editBuilder.insert(segmentPosition, `${commentChar} `);
                });
                segmentPosition = new vscode.Position(
                    segmentPosition.line,
                    commentChar.length + 1
                );
            }

            const line = segment.text;
            for (let i = 0; i < line.length; i += chunkSize) {
                if (!await this.validateEditorState(editor, pendingMessage)) {
                    return null;
                }

                const chunk = line.slice(i, Math.min(i + chunkSize, line.length));
                await editor.edit(editBuilder => {
                    editBuilder.insert(segmentPosition, chunk);
                });
                segmentPosition = new vscode.Position(
                    segmentPosition.line,
                    segmentPosition.character + chunk.length
                );
                await new Promise(resolve => setTimeout(resolve, 10));
            }

            await editor.edit(editBuilder => {
                editBuilder.insert(segmentPosition, '\n');
            });

            return new vscode.Position(segmentPosition.line + 1, 0);
        } catch (error) {
            console.error('Error processing segment:', error);
            return null;
        }
    }

    async appendHeader(
        editor: vscode.TextEditor,
        pendingMessage: PendingMessage,
        needsClosingTag: boolean,
        commentChar: string
    ): Promise<void> {
        pendingMessage.decorations = {
            blueMarker: vscode.window.createTextEditorDecorationType({
                color: BLUE_COLOR
            }),
            whiteText: vscode.window.createTextEditorDecorationType({
                color: WHITE_COLOR
            })
        };

        const headerStartPosition = new vscode.Position(pendingMessage.position.line, 0);
        if (needsClosingTag) {
            await editor.edit(editBuilder => {
                editBuilder.insert(headerStartPosition, `\n\n<!-- ${RESPONSE_START_MARKER}\n\n`);
            });
            pendingMessage.decorations.startRange = new vscode.Range(
                new vscode.Position(headerStartPosition.line + 2, 0),
                new vscode.Position(headerStartPosition.line + 2, 4 + RESPONSE_START_MARKER.length)
            );
            pendingMessage.editorState.headerPosition = new vscode.Position(headerStartPosition.line + 4, 0);
        } else {
            await editor.edit(editBuilder => {
                editBuilder.insert(headerStartPosition, `\n\n${commentChar} ${RESPONSE_START_MARKER}\n\n`);
            });
            pendingMessage.decorations.startRange = new vscode.Range(
                new vscode.Position(headerStartPosition.line + 2, 0),
                new vscode.Position(headerStartPosition.line + 2, commentChar.length + 1 + RESPONSE_START_MARKER.length)
            );
            pendingMessage.editorState.headerPosition = new vscode.Position(headerStartPosition.line + 4, 0);
        }

        editor.setDecorations(pendingMessage.decorations.blueMarker, [pendingMessage.decorations.startRange]);
        pendingMessage.editorState.headerWritten = true;
    }

    async appendFooter(
        editor: vscode.TextEditor,
        pendingMessage: PendingMessage,
        needsClosingTag: boolean,
        commentChar: string
    ): Promise<void> {
        const lastSegmentId = pendingMessage.editorState.segmentOrder[pendingMessage.editorState.segmentOrder.length - 1];
        const lastSegment = pendingMessage.segments.get(lastSegmentId)!;
        const position = lastSegment.writtenPosition || pendingMessage.position;

        try {
            if (needsClosingTag) {
                await editor.edit(editBuilder => {
                    editBuilder.insert(position, `\n\n${RESPONSE_END_MARKER} -->\n\n`);
                });
                pendingMessage.decorations!.endRange = new vscode.Range(
                    new vscode.Position(position.line + 1, 0),
                    new vscode.Position(position.line + 1, RESPONSE_END_MARKER.length + 5)
                );
            } else {
                await editor.edit(editBuilder => {
                    editBuilder.insert(position, `\n${commentChar} ${RESPONSE_END_MARKER}\n\n`);
                });
                pendingMessage.decorations!.endRange = new vscode.Range(
                    new vscode.Position(position.line + 1, 0),
                    new vscode.Position(position.line + 1, commentChar.length + 1 + RESPONSE_END_MARKER.length)
                );
            }

            this.updateDecorations(editor, pendingMessage);
        } catch (error) {
            console.error('Error in appendFooter:', error);
        }
    }

    private updateDecorations(editor: vscode.TextEditor, pendingMessage: PendingMessage): void {
        if (!pendingMessage.decorations?.startRange || !pendingMessage.decorations?.endRange) {return;}

        pendingMessage.decorations.contentRange = new vscode.Range(
            new vscode.Position(pendingMessage.decorations.startRange.end.line + 1, 0),
            new vscode.Position(pendingMessage.decorations.endRange.start.line, 0)
        );

        editor.setDecorations(pendingMessage.decorations.whiteText, [pendingMessage.decorations.contentRange]);
        editor.setDecorations(pendingMessage.decorations.blueMarker, [
            pendingMessage.decorations.startRange,
            pendingMessage.decorations.endRange
        ]);
    }

    async renderContent(editor: vscode.TextEditor, pendingMessage: PendingMessage, response: string, needsClosingTag: boolean, commentChar: string): Promise<void> {
        if (!await this.validateEditorState(editor, pendingMessage)) {
            return;
        }

        const formattedResponse = formatResponse(response);
        const [segments, segmentOrder] = this.segmentMessage(formattedResponse, pendingMessage.position);
        
        pendingMessage.segments = segments;
        pendingMessage.editorState.segmentOrder = segmentOrder;

        for (const segmentId of segmentOrder) {
            const segment = segments.get(segmentId)!;
            const position = await this.processSegment(
                editor,
                segment,
                segment.position,
                needsClosingTag,
                commentChar,
                pendingMessage
            );

            if (position) {
                segment.written = true;
                segment.writtenPosition = position;
                pendingMessage.editorState.lastProcessedSegmentId = segmentId;
            } else {
                break;
            }
        }

        pendingMessage.editorState.contentComplete = 
            pendingMessage.editorState.lastProcessedSegmentId === segmentOrder[segmentOrder.length - 1];
    }
}

export const messageStateManager = new MessageStateManager();

// Main response handler
interface ChunkState {
    text: string;
    isRendered: boolean;
    position: vscode.Position;
    lineContent?: string;
}

interface MessageRenderState {
    id: string;
    chunks: ChunkState[];
    currentChunkIndex: number;
    isComplete: boolean;
    headerPosition?: vscode.Position;
    documentUri: string;
    decorations: {
        blueMarker: vscode.TextEditorDecorationType;
        whiteText: vscode.TextEditorDecorationType;
    };
}

class MessageRenderer {
    private static CHUNK_SIZE = 50;
    private static RENDER_DELAY = 10;
    private static instance: MessageRenderer;
    private renderStates: Map<string, MessageRenderState> = new Map();
    private activeRendering: Set<string> = new Set();
    private isDisposed = false;

    private constructor() {
        // Handle editor changes
        vscode.window.onDidChangeActiveTextEditor(this.handleEditorChange.bind(this));
        
        // Cleanup on deactivation
        this.setupCleanup();
    }

    private setupCleanup() {
        // Ensure cleanup when extension is deactivated
        if (vscode.extensions.all.length > 0) {
            const extension = vscode.extensions.all[0];
            if (extension.activate) {
                extension.activate().then(() => {
                    extension.exports?.deactivate?.(() => {
                        this.dispose();
                    });
                });
            }
        }
    }

    static getInstance(): MessageRenderer {
        if (!MessageRenderer.instance) {
            MessageRenderer.instance = new MessageRenderer();
        }
        return MessageRenderer.instance;
    }

    private async handleEditorChange(editor: vscode.TextEditor | undefined) {
        if (!editor) {return;}

        const documentId = editor.document.uri.toString();
        const state = this.renderStates.get(documentId);

        if (state && !state.isComplete) {
            await this.validateAndResume(editor, documentId);
        }
    }

    private async validateAndResume(editor: vscode.TextEditor, documentId: string) {
        const state = this.renderStates.get(documentId);
        if (!state || this.activeRendering.has(documentId)) {return;}

        try {
            // Validate existing content before resuming
            const validationResult = await this.validateContent(editor, state);
            if (validationResult.isValid) {
                await this.resumeRendering(editor, documentId);
            } else {
                // Content is invalid - need to restart from last valid point
                await this.restartFromValidPoint(editor, state, validationResult.lastValidIndex);
            }
        } catch (error) {
            console.error('Error during validation and resume:', error);
            if (error instanceof Error && error.message.includes('closed editors')) {
                this.renderStates.delete(documentId);
            }
        }
    }

    private async validateContent(
        editor: vscode.TextEditor,
        state: MessageRenderState
    ): Promise<{ isValid: boolean; lastValidIndex: number }> {
        let lastValidIndex = -1;

        try {
            for (let i = 0; i < state.currentChunkIndex; i++) {
                const chunk = state.chunks[i];
                if (!chunk.isRendered) {continue;}

                const line = editor.document.lineAt(chunk.position.line);
                const expectedContent = chunk.lineContent || chunk.text;
                
                if (line.text.includes(expectedContent)) {
                    lastValidIndex = i;
                } else {
                    break;
                }
            }
        } catch (error) {
            console.error('Error during content validation:', error);
        }

        return {
            isValid: lastValidIndex === state.currentChunkIndex - 1,
            lastValidIndex: Math.max(0, lastValidIndex)
        };
    }

    private async restartFromValidPoint(
        editor: vscode.TextEditor,
        state: MessageRenderState,
        lastValidIndex: number
    ) {
        // Reset state to last valid point
        state.currentChunkIndex = lastValidIndex;
        
        // Mark subsequent chunks as not rendered
        for (let i = lastValidIndex + 1; i < state.chunks.length; i++) {
            state.chunks[i].isRendered = false;
        }

        // Resume rendering
        await this.resumeRendering(editor, editor.document.uri.toString());
    }

    private async resumeRendering(editor: vscode.TextEditor, documentId: string) {
        const state = this.renderStates.get(documentId);
        if (!state || this.activeRendering.has(documentId)) {return;}

        this.activeRendering.add(documentId);

        try {
            for (let i = state.currentChunkIndex; i < state.chunks.length; i++) {
                const chunk = state.chunks[i];
                if (this.isDisposed || !this.isEditorValid(editor)) {
                    this.activeRendering.delete(documentId);
                    return;
                }

                if (!chunk.isRendered) {
                    const success = await this.renderChunkSafely(editor, chunk, state);
                    if (!success) {
                        this.activeRendering.delete(documentId);
                        return;
                    }
                    state.currentChunkIndex = i + 1;
                    await new Promise(resolve => setTimeout(resolve, MessageRenderer.RENDER_DELAY));
                }
            }

            if (state.currentChunkIndex >= state.chunks.length) {
                state.isComplete = true;
                await this.appendFooterSafely(editor, state);
            }
        } finally {
            this.activeRendering.delete(documentId);
        }
    }

    private isEditorValid(editor: vscode.TextEditor): boolean {
        return !!(editor && 
                 editor.document && 
                 !editor.document.isClosed && 
                 editor === vscode.window.activeTextEditor);
    }

    private async renderChunkSafely(
        editor: vscode.TextEditor,
        chunk: ChunkState,
        state: MessageRenderState
    ): Promise<boolean> {
        try {
            if (!this.isEditorValid(editor)) {return false;}

            await editor.edit(editBuilder => {
                editBuilder.insert(chunk.position, chunk.text);
            });
            
            chunk.isRendered = true;
            chunk.lineContent = editor.document.lineAt(chunk.position.line).text;
            return true;
        } catch (error) {
            console.error('Error rendering chunk:', error);
            return false;
        }
    }

    private async appendFooterSafely(
        editor: vscode.TextEditor,
        state: MessageRenderState
    ) {
        try {
            if (!this.isEditorValid(editor)) {return;}

            const lastChunk = state.chunks[state.chunks.length - 1];
            await editor.edit(editBuilder => {
                editBuilder.insert(lastChunk.position, '\n');
            });

            this.updateDecorations(editor, state);
        } catch (error) {
            console.error('Error appending footer:', error);
        }
    }

    async startRendering(
        editor: vscode.TextEditor,
        response: string,
        startPosition: vscode.Position
    ): Promise<void> {
        const documentId = editor.document.uri.toString();
        
        // Create new render state
        const state: MessageRenderState = {
            id: `msg-${Date.now()}`,
            chunks: this.createChunks(response, startPosition),
            currentChunkIndex: 0,
            isComplete: false,
            documentUri: documentId,
            decorations: {
                blueMarker: vscode.window.createTextEditorDecorationType({
                    color: new vscode.ThemeColor('editorInfo.foreground')
                }),
                whiteText: vscode.window.createTextEditorDecorationType({
                    color: new vscode.ThemeColor('editor.foreground')
                })
            }
        };

        this.renderStates.set(documentId, state);
        await this.resumeRendering(editor, documentId);
    }

    private createChunks(text: string, startPosition: vscode.Position): ChunkState[] {
        const lines = text.split('\n');
        const chunks: ChunkState[] = [];
        let currentPosition = startPosition;

        for (const line of lines) {
            chunks.push({
                text: line + '\n',
                isRendered: false,
                position: currentPosition
            });
            currentPosition = new vscode.Position(currentPosition.line + 1, 0);
        }

        return chunks;
    }

    private updateDecorations(editor: vscode.TextEditor, state: MessageRenderState) {
        if (!this.isEditorValid(editor)) {return;}

        const startLine = state.chunks[0].position.line;
        const endLine = state.chunks[state.chunks.length - 1].position.line;
        
        const contentRange = new vscode.Range(
            new vscode.Position(startLine, 0),
            new vscode.Position(endLine, Number.MAX_VALUE)
        );

        editor.setDecorations(state.decorations.whiteText, [contentRange]);
    }

    dispose() {
        this.isDisposed = true;
        for (const state of this.renderStates.values()) {
            state.decorations.blueMarker.dispose();
            state.decorations.whiteText.dispose();
        }
        this.renderStates.clear();
        this.activeRendering.clear();
    }

    cleanup(documentId: string) {
        const state = this.renderStates.get(documentId);
        if (state) {
            state.decorations.blueMarker.dispose();
            state.decorations.whiteText.dispose();
            this.renderStates.delete(documentId);
        }
    }
}

// Updated appendResponseToFile function
async function appendResponseToFile(
    editor: vscode.TextEditor,
    response: string,
    animationControl: AnimationControl,
    fileType: string
): Promise<void> {
    try {
        await editor.edit(editBuilder => {
            editBuilder.delete(animationControl.animationRange);
        });

        const renderer = MessageRenderer.getInstance();
        await renderer.startRendering(
            editor,
            formatResponse(response),
            new vscode.Position(animationControl.insertPosition.line, 0)
        );
    } catch (error) {
        console.error('Error in appendResponseToFile:', error);
        vscode.window.showErrorMessage('Failed to render response. Please try again.');
    }
}


function formatResponse(response: string | undefined): string {
    if (!response) {
        console.error('Received undefined response in formatResponse');
        return 'Error: No response received from the model.';
    }
    // Split the response into lines
    const lines = response.split('\n');
    // Remove any empty lines at the start and end
    while (lines.length > 0 && lines[0].trim() === '')
        {lines.shift();}
    while (lines.length > 0 && lines[lines.length - 1].trim() === '')
        {lines.pop();}
    // Join the lines back together
    return lines.join('\n');
}

async function extractCodeFromLastResponse() {
    const editor = window.activeTextEditor;
    const notebookEditor = vscode.window.activeNotebookEditor;
    if (notebookEditor) {
        await extractCodeFromNotebook(notebookEditor);
        return;
    }
    if (!editor) {
        window.showInformationMessage('No active editor found.');
        return;
    }
    if (storedCodeBlocks.length > 0) {
        // If we have stored code blocks, show the picker
        await showCodeBlockPicker(editor);
        return;
    }
    const document = editor.document;
    const fullText = document.getText();
    // Find the last response
    const lastResponseStartIndex = fullText.lastIndexOf(RESPONSE_START_MARKER);
    const lastResponseEndIndex = fullText.lastIndexOf(RESPONSE_END_MARKER);
    if (lastResponseStartIndex === -1 || lastResponseEndIndex === -1 || lastResponseStartIndex > lastResponseEndIndex) {
        window.showInformationMessage('No valid response found in the document.');
        return;
    }
    const responseText = fullText.substring(lastResponseStartIndex, lastResponseEndIndex + RESPONSE_END_MARKER.length);
    // Extract code blocks
    const codeBlockRegex = /```[\s\S]*?```/g;
    const codeBlocks = responseText.match(codeBlockRegex);
    if (!codeBlocks || codeBlocks.length === 0) {
        window.showInformationMessage('No code blocks found in the last response.');
        return;
    }
    // Extract code and language for each code block
    storedCodeBlocks = codeBlocks.map(extractCodeAndLanguage);
    // Store the response range for later removal
    lastResponseRange = new Range(document.positionAt(lastResponseStartIndex), document.positionAt(lastResponseEndIndex + RESPONSE_END_MARKER.length));
    if (storedCodeBlocks.length <= 3) {
        // If 3 or fewer blocks, insert them all without prompting
        await removeLastResponse(editor);
        for (const block of storedCodeBlocks) {
            await insertCodeBlock(editor, block);
        }
        window.showInformationMessage('All code blocks have been inserted.');
        storedCodeBlocks = []; // Clear stored blocks after insertion
    }
    else {
        // If more than 3 blocks, use the picker
        await showCodeBlockPicker(editor);
    }
}

async function showCodeBlockPicker(editorOrNotebook: vscode.TextEditor | vscode.NotebookEditor) {
    if (storedCodeBlocks.length === 0) {
        vscode.window.showInformationMessage('No stored code blocks available.');
        return;
    }
    const quickPickItems = storedCodeBlocks.map((block, index) => ({
        label: `Code Block ${index + 1}${block.language ? ` (${block.language})` : ''}`,
        description: block.code.split('\n')[0].substring(0, 50) + '...',
        block: block
    }));
    const selectedItem = await vscode.window.showQuickPick(quickPickItems, {
        placeHolder: 'Select a code block to insert'
    });
    if (selectedItem && selectedItem.block) {
        if ('document' in editorOrNotebook) {
            // This is a TextEditor
            if (lastResponseRange) {
                await removeLastResponse(editorOrNotebook);
                lastResponseRange = null;
            }
            await insertCodeBlock(editorOrNotebook, selectedItem.block);
        }
        else if ('notebook' in editorOrNotebook) {
            // This is a NotebookEditor
            await insertCodeBlockToNotebook(editorOrNotebook, selectedItem.block);
        }
    }
    if (storedCodeBlocks.length > 0) {
        vscode.window.showInformationMessage('You can insert more blocks by running the extract code command again.');
    }
    else {
        vscode.window.showInformationMessage('All code blocks have been inserted.');
    }
}

async function removeLastResponse(editor: TextEditor) {
    if (lastResponseRange) {
        await editor.edit(editBuilder => {
            editBuilder.delete(lastResponseRange!);
        });
    }
}

async function insertCodeBlock(editor: TextEditor, codeBlock: {
    code: string;
    language: string | null;
}) {
    const insertPosition = findInsertPosition(editor);
    await editor.edit(editBuilder => {
        // Add a new empty line before the code block
        editBuilder.insert(insertPosition, '\n' + codeBlock.code + '\n\n');
    });
    window.showInformationMessage(`Code block inserted. ${codeBlock.language ? `Language: ${codeBlock.language}` : 'No specific language detected.'}`);
}

function findInsertPosition(editor: TextEditor): Position {
    const document = editor.document;
    const currentPosition = editor.selection.active;
    // Start from the current line and move down
    for (let line = currentPosition.line; line < document.lineCount; line++) {
        if (document.lineAt(line).isEmptyOrWhitespace) {
            return new Position(line, 0);
        }
    }
    // If no empty line found, insert at the end of the document
    return new Position(document.lineCount, 0);
}

function extractCodeAndLanguage(codeBlock: string): {
    code: string;
    language: string | null;
} {
    // Remove the opening and closing backticks
    let code = codeBlock.replace(/^```/, '').replace(/```$/, '').trim();
    let language: string | null = null;
    // Check if there's a language specified
    const firstLineBreak = code.indexOf('\n');
    if (firstLineBreak !== -1) {
        const firstLine = code.substring(0, firstLineBreak).trim();
        if (firstLine && !firstLine.includes(' ')) {
            language = firstLine;
            code = code.substring(firstLineBreak + 1).trim();
        }
    }
    else {
        // If there's no line break, check if the entire content is a language identifier
        if (code && !code.includes(' ')) {
            language = code;
            code = '';
        }
    }
    return { code, language };
}

function highlightPromptMarker(editor: vscode.TextEditor) {
	const document = editor.document;
	const fileType = editor.document.languageId;
  
	// For Python files, only look for the prompt marker at the beginning of lines
	if (fileType === 'python') {
	  for (let i = 0; i < document.lineCount; i++) {
		const line = document.lineAt(i);
		const lineText = line.text.trimStart();
		
		if (lineText.startsWith(PROMPT_MARKER)) {
		  const startPos = new vscode.Position(i, line.firstNonWhitespaceCharacterIndex);
		  const endPos = new vscode.Position(i, line.firstNonWhitespaceCharacterIndex + PROMPT_MARKER.length);
		  const range = new vscode.Range(startPos, endPos);
		  editor.setDecorations(promptDecoration, [range]);
		  return;
		}
	  }
	} else {
	  // Existing logic for other file types
	  for (let i = 0; i < document.lineCount; i++) {
		const line = document.lineAt(i);
		const lineText = line.text;
		const markerIndex = lineText.lastIndexOf(PROMPT_MARKER);
  
		if (markerIndex !== -1 && !lineText.trim().endsWith('-->')) {
		  const range = new vscode.Range(
			new vscode.Position(i, markerIndex),
			new vscode.Position(i, markerIndex + PROMPT_MARKER.length)
		  );
		  editor.setDecorations(promptDecoration, [range]);
		  return;
		}
	  }
	}
  
	// If we reach here, the marker was not found
	isPromptMode = false;
	editor.setDecorations(promptDecoration, []);
  }

  async function handleTextDocumentChange(event: vscode.TextDocumentChangeEvent) {
	const editor = vscode.window.activeTextEditor;
	if (editor && editor.document === event.document && isPromptMode) {
	  if (doesPromptMarkerExist(editor)) {
		highlightPromptMarker(editor);
	  } else {
		isPromptMode = false;
		editor.setDecorations(promptDecoration, []);
	  }
	}
  }
  
  function highlightCommandMarker(editor: vscode.TextEditor) {
	const document = editor.document;
	for (let i = 0; i < document.lineCount; i++) {
	  const line = document.lineAt(i);
	  const lineText = line.text;
	  const markerIndex = lineText.lastIndexOf('?');
  
	  if (markerIndex !== -1) {
		const range = new vscode.Range(
		  new vscode.Position(i, markerIndex),
		  new vscode.Position(i, markerIndex + 1)
		);
		editor.setDecorations(commandDecoration, [range]);
		return;
	  }
	}
  
	isCommandMode = false;
	editor.setDecorations(commandDecoration, []);
  }


  async function handlePromptEnter(editor: vscode.TextEditor) {
	const document = editor.document;
	const fileType = document.languageId;
	const line = document.lineAt(editor.selection.active.line);
	const lineText = line.text;
	const commandMarkerIndex = lineText.lastIndexOf('?');

	if (commandMarkerIndex !== -1) {
		const commandText = lineText.substring(commandMarkerIndex + 1).trim();
		if (commandText) {
		  await handleCommand(commandText);
		  // Clear the command line after sending
		  await editor.edit(editBuilder => {
			editBuilder.delete(line.range);
		  });
		}
		isCommandMode = false;
		editor.setDecorations(commandDecoration, []);
		return null; // Prevent default Enter behavior
	  }

	  if (fileType === 'python') {
	  const trimmedLineText = lineText.trimStart();
	  if (trimmedLineText.startsWith(PROMPT_MARKER)) {
		const promptText = trimmedLineText.substring(PROMPT_MARKER.length).trim();
		if (promptText) {
		  await handleChat(promptText);
		  // Clear the prompt line after sending
		  await editor.edit(editBuilder => {
			editBuilder.delete(line.range);
		  });
		}
		isPromptMode = false;
		editor.setDecorations(promptDecoration, []);
		return null; // Prevent default Enter behavior
	  }
	} else {
	  // Existing logic for other file types
	  const markerIndex = lineText.lastIndexOf(PROMPT_MARKER);
  
	  if (markerIndex !== -1 && !lineText.trim().endsWith('-->')) {
		const promptText = lineText.substring(markerIndex + 1).trim();
		if (promptText) {
		  await handleChat(promptText);
		  // Clear the prompt line after sending
		  await editor.edit(editBuilder => {
			editBuilder.delete(line.range);
		  });
		}
		isPromptMode = false;
		editor.setDecorations(promptDecoration, []);
		return null; // Prevent default Enter behavior
	  }
	}
  
	return vscode.commands.executeCommand('default:type', { text: '\n' });
  }


export {
    getAvailableModels, getLocalModels, updateModelList, doesPromptMarkerExist,
    highlightPromptMarker, handleTextDocumentChange, highlightCommandMarker,
    handlePromptEnter, handleStatusCommand, handleShutdownCommand,
    startRequestAnimation, detectTracebackError, handleNotebookDocumentChange,
    findLastExecutedCell, startNotebookLoadingAnimation, handleChatNotebook,
    getNormalCellSelection, extractCodeFromNotebook, insertCodeBlockToNotebook,
    appendResponseToNotebook, splitResponseIntoParts, handleChat,
    getInsertPosition, getPrompt, getCurrentModel, sendMessage,
    sendLocalModelMessage, sendOpenAIMessage, sendAnthropicMessage,
    sendGeminiMessage, appendResponseToFile, formatResponse,
    extractCodeFromLastResponse, showCodeBlockPicker, removeLastResponse,
    insertCodeBlock, findInsertPosition, extractCodeAndLanguage};