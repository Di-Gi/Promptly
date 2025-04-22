// src/editor/markerHandler.ts
import * as vscode from 'vscode';
import { TextEditor, Position, Range, TextDocumentChangeEvent } from 'vscode';
import { PROMPT_MARKER, COMMAND_MARKER, GREEN_COLOR, BLUE_COLOR } from '../common/constants';
import { handleChat } from '../chat/chatService'; // Adjusted import
import { handleServerCommand } from '../server/serverService'; // Adjusted import

let promptDecoration: vscode.TextEditorDecorationType | undefined;
let commandDecoration: vscode.TextEditorDecorationType | undefined;

/**
 * Initializes the decorations used for prompt and command markers.
 * Should be called once during extension activation.
 */
export function initializeDecorations() {
    promptDecoration = vscode.window.createTextEditorDecorationType({
        // More distinct styling for prompt marker
        // color: GREEN_COLOR,
        // fontWeight: 'bold',
        backgroundColor: new vscode.ThemeColor('editor.selectionHighlightBackground'),
        borderRadius: '3px',
        overviewRulerColor: 'green',
        overviewRulerLane: vscode.OverviewRulerLane.Left,
         before: {
             contentText: '▶',
             color: GREEN_COLOR,
             margin: '0 6px 0 0',
         }
    });
    commandDecoration = vscode.window.createTextEditorDecorationType({
        // More distinct styling for command marker
        // color: BLUE_COLOR,
        // fontWeight: 'bold',
         backgroundColor: new vscode.ThemeColor('editor.findMatchHighlightBackground'),
         borderRadius: '3px',
        overviewRulerColor: 'blue',
        overviewRulerLane: vscode.OverviewRulerLane.Left,
          before: {
             contentText: '⚡',
             color: BLUE_COLOR,
             margin: '0 6px 0 0',
         }
    });
}

/**
 * Disposes of the decorations.
 * Should be called during extension deactivation.
 */
export function disposeDecorations() {
    promptDecoration?.dispose();
    commandDecoration?.dispose();
    promptDecoration = undefined;
    commandDecoration = undefined;
}

/**
 * Highlights the prompt marker (>>) in the editor, considering file type.
 */
function highlightPromptMarker(editor: TextEditor) {
    if (!promptDecoration) {return;}

    const document = editor.document;
    const promptRanges: Range[] = [];
    const fileType = editor.document.languageId; // Use languageId for consistency

    for (let i = 0; i < document.lineCount; i++) {
        const line = document.lineAt(i);
        const lineText = line.text;
        const trimmedLineText = lineText.trimStart();
        const markerLength = PROMPT_MARKER.length;

        // Python: Only at the start of the line (after whitespace)
        if (fileType === 'python') {
            if (trimmedLineText.startsWith(PROMPT_MARKER)) {
                const startCharIndex = line.firstNonWhitespaceCharacterIndex;
                const range = new Range(
                    new Position(i, startCharIndex),
                    new Position(i, startCharIndex + markerLength)
                );
                promptRanges.push(range);
            }
        } else {
            // Other file types: Check for marker, avoid comment terminators
            // We might need more robust comment detection depending on languages
            const markerIndex = lineText.indexOf(PROMPT_MARKER); // Use indexOf for first occurrence maybe? or lastIndexOf? Stick with last for now.
            if (markerIndex !== -1 && !lineText.substring(markerIndex).includes('-->')) {
                // Add heuristic: is it likely intended as a prompt? (e.g., near start of line, or after comment prefix?)
                // Simple check: is it within the first few chars after trimming?
                 if (markerIndex <= line.firstNonWhitespaceCharacterIndex + 5) {
                      const range = new Range(
                          new Position(i, markerIndex),
                          new Position(i, markerIndex + markerLength)
                      );
                      promptRanges.push(range);
                 }
            }
        }
    }
    editor.setDecorations(promptDecoration, promptRanges);
}

/**
 * Highlights the command marker (??) in the editor.
 * Only highlights if it's at the beginning of the line (after whitespace).
 */
function highlightCommandMarker(editor: TextEditor) {
    if (!commandDecoration) {return;}

    const document = editor.document;
    const commandRanges: Range[] = [];
    const markerLength = COMMAND_MARKER.length;

    for (let i = 0; i < document.lineCount; i++) {
        const line = document.lineAt(i);
        const trimmedLineText = line.text.trimStart();

        if (trimmedLineText.startsWith(COMMAND_MARKER)) {
            const startCharIndex = line.firstNonWhitespaceCharacterIndex;
            const range = new Range(
                new Position(i, startCharIndex),
                new Position(i, startCharIndex + markerLength)
            );
            commandRanges.push(range);
            // break; // Optional: Highlight only the first command marker found? Usually allow multiple.
        }
    }
    editor.setDecorations(commandDecoration, commandRanges);
}

/**
 * Checks if a prompt marker exists anywhere in the document (basic check).
 */
export function doesPromptMarkerExist(editor: TextEditor): boolean {
    // This is a simplified check, could be refined using the highlighting logic if needed.
    const document = editor.document;
    const text = document.getText();
    // Basic check, might catch markers inside comments in non-python files
    return text.includes(PROMPT_MARKER);
}


/**
 * Handles text document changes to update marker highlights.
 */
export function handleTextDocumentChange(event: TextDocumentChangeEvent) {
    const editor = vscode.window.activeTextEditor;
    // Only proceed if the change happened in the currently active editor
    if (editor && editor.document === event.document) {
        // Re-apply highlights based on the current document state
        highlightPromptMarker(editor);
        highlightCommandMarker(editor);
    }
}

/**
 * Handles the Enter key press in the editor.
 * Checks if the current line is a command or prompt and executes accordingly.
 * Returns null to suppress default Enter behavior if a command/prompt was handled.
 */
export async function handlePromptEnter(editor: TextEditor): Promise<unknown | null> {
    const document = editor.document;
    const currentLineNumber = editor.selection.active.line;

     // Ensure line number is valid (might be outdated if doc changed rapidly)
     if (currentLineNumber >= document.lineCount) {
         return vscode.commands.executeCommand('default:type', { text: '\n' }); // Default action
     }

    const line = document.lineAt(currentLineNumber);
    const lineText = line.text;
    const trimmedLineText = lineText.trimStart();
    const fileType = document.languageId; // Get file type for context

    // 1. Check for Command Marker (??) at the start of the line
    if (trimmedLineText.startsWith(COMMAND_MARKER)) {
        const commandText = trimmedLineText.substring(COMMAND_MARKER.length).trim();
        // Clear the command line *before* executing the command (feels more responsive)
        await editor.edit(editBuilder => {
            editBuilder.delete(line.range);
        });
        if (commandText) {
            await handleServerCommand(commandText); // Delegate to server service
        } else {
            vscode.window.showInformationMessage("Command cannot be empty.");
        }
        // Explicitly clear decorations immediately after handling
        if(commandDecoration) {editor.setDecorations(commandDecoration, []);}
        return null; // Suppress default Enter
    }

    // 2. Check for Prompt Marker (>>)
    let isPromptLine = false;
    let promptText = '';
    const promptMarkerLength = PROMPT_MARKER.length;

    // Python: Check start of trimmed line
    if (fileType === 'python') {
        if (trimmedLineText.startsWith(PROMPT_MARKER)) {
            isPromptLine = true;
            promptText = trimmedLineText.substring(promptMarkerLength).trim();
        }
    } else {
        // Other files: Use heuristic (found marker near start of content)
        const markerIndex = lineText.indexOf(PROMPT_MARKER);
        if (markerIndex !== -1 && !lineText.substring(markerIndex).includes('-->')) {
            if (markerIndex <= line.firstNonWhitespaceCharacterIndex + 5) {
                 isPromptLine = true;
                 promptText = lineText.substring(markerIndex + promptMarkerLength).trim();
             }
        }
    }

    if (isPromptLine) {
         // Clear the prompt line *before* sending the request
         await editor.edit(editBuilder => {
             editBuilder.delete(line.range);
         });
        if (promptText) {
            await handleChat(promptText); // Pass only the text after the marker
        } else {
            vscode.window.showInformationMessage("Prompt cannot be empty.");
            // Maybe trigger default handleChat without override? Optional.
            // await handleChat();
        }
        // Explicitly clear decorations immediately after handling
        if(promptDecoration) {editor.setDecorations(promptDecoration, []);}
        return null; // Suppress default Enter
    }

    // 3. Default Action: If neither command nor prompt, execute default Enter behavior
    return vscode.commands.executeCommand('default:type', { text: '\n' });
}