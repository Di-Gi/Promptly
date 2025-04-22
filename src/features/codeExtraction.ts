// src/features/codeExtraction.ts
import * as vscode from 'vscode';
import { TextEditor, NotebookEditor, Range, Position, NotebookCellKind } from 'vscode';
import { CodeBlock } from '../common/types';
import { findInsertPositionForCodeBlock } from '../editor/editorService'; // Use editor service helper
import { insertCodeBlockToNotebook } from '../notebook/notebookUtils'; // Use notebook util


// --- State Management ---
// This state needs to be managed carefully. A class might be better if complexity grows.
// For now, keep module-level state. Keys could be document URIs if managing multiple docs.
let storedCodeBlocks: CodeBlock[] = [];
let lastResponseRange: Range | null = null; // Store the range of the *original AI response* in the text editor

// --- Helper Functions ---

/**
 * Extracts code content and language identifier from a Markdown code block string.
 */
function extractCodeAndLanguage(codeBlockString: string): CodeBlock {
     // Regex to capture optional language and the code content
     const match = codeBlockString.match(/^```(\w+)?\s*\n?([\s\S]*?)```$/);
     if (match) {
         const language = match[1] || null; // Language identifier (e.g., 'python', 'javascript') or null
         const code = match[2].trim();    // The actual code content
         return { code, language };
     } else {
         // Fallback if regex fails (e.g., malformed block)
         const code = codeBlockString.replace(/^```/, '').replace(/```$/, '').trim();
         return { code, language: null };
     }
 }


/**
 * Removes the previously identified AI response range from the text editor.
 */
async function removeLastResponseFromEditor(editor: TextEditor) {
    if (lastResponseRange) {
        try {
             await editor.edit(editBuilder => {
                 editBuilder.delete(lastResponseRange!);
             });
             lastResponseRange = null; // Clear the range after successful removal
        } catch (error) {
             console.error("Failed to remove last response range:", error);
             vscode.window.showErrorMessage("Could not remove the previous AI response.");
             // Should we clear lastResponseRange here? Maybe not, to allow retry?
         }
    }
}

/**
 * Inserts a code block into the text editor at a suitable position.
 */
async function insertCodeBlockToEditor(editor: TextEditor, codeBlock: CodeBlock) {
    const insertPosition = findInsertPositionForCodeBlock(editor);
    try {
         await editor.edit(editBuilder => {
             // Add newlines before and after for better spacing, unless inserting at start/end of doc?
             const textToInsert = `\n${codeBlock.code}\n`; // Simpler insertion
             editBuilder.insert(insertPosition, textToInsert);
         });
         console.log(`Code block (${codeBlock.language || 'text'}) inserted into editor.`);
         // Optional: Format the inserted code block if language is known?
         // vscode.commands.executeCommand('editor.action.formatDocument'); // Might format too much
    } catch (error) {
         console.error("Failed to insert code block into editor:", error);
         vscode.window.showErrorMessage(`Failed to insert code block: ${error}`);
     }
}


// --- Core Logic Functions ---

/**
 * Shows a Quick Pick menu to let the user select a code block to insert.
 */
async function showCodeBlockPicker(editorOrNotebook: TextEditor | NotebookEditor) {
    if (storedCodeBlocks.length === 0) {
        vscode.window.showInformationMessage('No extracted code blocks available.');
        return;
    }

    const quickPickItems = storedCodeBlocks.map((block, index) => ({
        label: `Code Block ${index + 1}${block.language ? ` (${block.language})` : ''}`,
        description: block.code.split('\n')[0]?.substring(0, 80) + '...' || '[Empty Block]', // Handle empty blocks
        detail: `Lines: ${block.code.split('\n').length}`,
        block: block,
        index: index // Store original index
    }));

    const selectedItem = await vscode.window.showQuickPick(quickPickItems, {
        placeHolder: 'Select a code block to insert',
        canPickMany: false // Allow inserting only one at a time via picker
    });

    if (selectedItem?.block) {
        // Remove the selected block from storage *before* inserting
         storedCodeBlocks.splice(selectedItem.index, 1);

        if (editorOrNotebook.hasOwnProperty('document')) { // It's a TextEditor
            const editor = editorOrNotebook as TextEditor;
            // Remove the original AI response *only when the first block is inserted* via picker
             if (lastResponseRange) {
                 await removeLastResponseFromEditor(editor);
                 // lastResponseRange is cleared by the removal function
             }
            await insertCodeBlockToEditor(editor, selectedItem.block);
        }
        else if (editorOrNotebook.hasOwnProperty('notebook')) { // It's a NotebookEditor
            await insertCodeBlockToNotebook(editorOrNotebook as NotebookEditor, selectedItem.block);
        }

        // Inform user if more blocks remain
        if (storedCodeBlocks.length > 0) {
            vscode.window.showInformationMessage(`Code block inserted. ${storedCodeBlocks.length} more block(s) available. Run 'Extract Code' again to choose another.`);
        } else {
            vscode.window.showInformationMessage('Last code block inserted.');
        }
    }
 }

/**
 * Extracts code blocks from the last AI response in the active TextEditor.
 * Stores them and either inserts directly or prompts the user with a picker.
 */
export async function extractCodeFromEditorResponse() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showInformationMessage('No active text editor found.');
        return;
    }

     // If blocks are already stored, just show the picker again
     if (storedCodeBlocks.length > 0) {
         console.log("Stored code blocks exist, showing picker.");
         await showCodeBlockPicker(editor);
         return;
     }

    // Reset state for new extraction
    storedCodeBlocks = [];
    lastResponseRange = null;

    const document = editor.document;
    const fullText = document.getText();

     // Heuristic to find the "last response" - This is fragile!
     // Requires the rendering process to somehow mark the response boundaries.
     // Let's assume MessageRenderer *could* store the range, or we find based on markers/heuristics.
     // Placeholder: Find based on last block of non-commented text? Very difficult.
     // **Simplification:** Assume the user runs this command *immediately* after a response is rendered.
     // We need a way to get the range of the last rendered response.
     // Let's modify MessageRenderer to store this or use a marker-based approach if needed.
     // **Temporary Workaround:** Find the last ``` block. This is not reliable.
    const codeBlockRegex = /```[\s\S]*?```/g;
    let match;
    let lastMatchIndex = -1;
    let lastMatchEndIndex = -1;
     let matches = [];
     while ((match = codeBlockRegex.exec(fullText)) !== null) {
         matches.push(match);
         lastMatchIndex = match.index;
         lastMatchEndIndex = match.index + match[0].length;
     }


    if (matches.length === 0) {
        vscode.window.showInformationMessage('No code blocks found in the document.');
        return;
    }

     // Assume the "response" is the text containing the last code block up to the end? Still weak.
     // TODO: Need a robust way to identify the response range.
     // Using last match index as a *very rough* proxy for response start.
     const potentialResponseStart = lastMatchIndex > 0 ? document.positionAt(lastMatchIndex) : new Position(0,0);
     const potentialResponseEnd = document.positionAt(fullText.length); // End of document
     const potentialResponseRange = new Range(potentialResponseStart, potentialResponseEnd); // Very broad guess

     const responseText = document.getText(potentialResponseRange);
     const codeBlocksInResponse = responseText.match(codeBlockRegex);


    if (!codeBlocksInResponse || codeBlocksInResponse.length === 0) {
        vscode.window.showInformationMessage('No code blocks found in the likely response area.');
        return;
    }

    storedCodeBlocks = codeBlocksInResponse.map(extractCodeAndLanguage);
    lastResponseRange = potentialResponseRange; // Store the *guessed* range

    console.log(`Extracted ${storedCodeBlocks.length} code blocks.`);

    if (storedCodeBlocks.length === 1) {
         // If only one block, insert it directly and remove the response.
         await removeLastResponseFromEditor(editor);
         await insertCodeBlockToEditor(editor, storedCodeBlocks[0]);
         storedCodeBlocks = []; // Clear storage
         vscode.window.showInformationMessage('Code block inserted.');
     } else if (storedCodeBlocks.length > 1) {
         // If multiple blocks, show the picker. Response removal happens when first block is chosen.
         await showCodeBlockPicker(editor);
     }
}

/**
 * Extracts code blocks from the last Markdown cell in the active NotebookEditor.
 * Stores them and either inserts directly or prompts the user with a picker.
 */
export async function extractCodeFromNotebookResponse(notebookEditor?: NotebookEditor) {
    if (!notebookEditor) {
         notebookEditor = vscode.window.activeNotebookEditor;
    }
     if (!notebookEditor) {
         vscode.window.showInformationMessage('No active notebook editor found.');
         return;
     }

     // If blocks are already stored, show picker
     if (storedCodeBlocks.length > 0) {
         console.log("Stored code blocks exist, showing picker for notebook.");
         await showCodeBlockPicker(notebookEditor);
         return;
     }


    const notebook = notebookEditor.notebook;
    const cells = notebook.getCells();

    // Find the last Markdown cell
    let lastMarkdownCell: vscode.NotebookCell | undefined;
    for (let i = cells.length - 1; i >= 0; i--) {
        if (cells[i].kind === NotebookCellKind.Markup) {
            lastMarkdownCell = cells[i];
            break;
        }
    }

    if (!lastMarkdownCell) {
        vscode.window.showInformationMessage('No Markdown cell found in the notebook to extract code from.');
        return;
    }

    const markdownContent = lastMarkdownCell.document.getText();
    const codeBlockRegex = /```[\s\S]*?```/g; // Same regex as editor
    const codeBlocksInMarkdown = markdownContent.match(codeBlockRegex);

    if (!codeBlocksInMarkdown || codeBlocksInMarkdown.length === 0) {
        vscode.window.showInformationMessage('No code blocks found in the last Markdown cell.');
        return;
    }

    // Reset state and store extracted blocks
    storedCodeBlocks = codeBlocksInMarkdown.map(extractCodeAndLanguage);
    lastResponseRange = null; // Not applicable for notebooks in the same way

     console.log(`Extracted ${storedCodeBlocks.length} code blocks from notebook markdown cell.`);

     // Similar logic: insert directly if 1, show picker if multiple
     if (storedCodeBlocks.length === 1) {
         // Option: Remove the source markdown cell? Maybe not, user might want it.
         await insertCodeBlockToNotebook(notebookEditor, storedCodeBlocks[0]);
         storedCodeBlocks = []; // Clear storage
         vscode.window.showInformationMessage('Code block inserted into new cell.');
     } else if (storedCodeBlocks.length > 1) {
         await showCodeBlockPicker(notebookEditor);
     }
}


/**
 * Entry point command to extract code, routes based on active editor type.
 */
export async function extractCodeCommand() {
    const activeEditor = vscode.window.activeTextEditor;
    const activeNotebook = vscode.window.activeNotebookEditor;

    if (activeNotebook) {
        console.log("Extracting code from Notebook context.");
        await extractCodeFromNotebookResponse(activeNotebook);
    } else if (activeEditor) {
        console.log("Extracting code from Text Editor context.");
        await extractCodeFromEditorResponse();
    } else {
        vscode.window.showInformationMessage('No active text editor or notebook found.');
    }
}
