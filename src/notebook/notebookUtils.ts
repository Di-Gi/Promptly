// src/notebook/notebookUtils.ts
import * as vscode from 'vscode';
import { NotebookDocument, NotebookCell, NotebookEditor, NotebookCellKind, NotebookCellData, WorkspaceEdit, NotebookRange, NotebookEdit } from 'vscode';
import { ResponsePart, CodeBlock } from '../common/types'; // Assuming types are shared

/**
 * Detects traceback errors in the output of a code cell.
 */
export function detectTracebackError(cell: NotebookCell): string | undefined { // <--- Changed return type
    if (cell.kind !== NotebookCellKind.Code || !cell.outputs || cell.outputs.length === 0) {
        return undefined;
    }

    for (const output of cell.outputs) {
        for (const item of output.items) {
            // Standard error MIME type
            if (item.mime === 'application/vnd.code.notebook.error') {
                try {
                    const errorData = JSON.parse(Buffer.from(item.data).toString('utf8'));
                    const tracebackText = `${errorData.ename || 'Error'}: ${errorData.evalue || ''}\n${(errorData.traceback || []).join('\n')}`;
                    return tracebackText.trim();
                } catch (e) {
                    console.error("Failed to parse notebook error data:", e);
                    return Buffer.from(item.data).toString('utf8');
                }
            }
            // Check other common MIME types
            else if (item.mime.startsWith('text/') || item.mime.includes('stderr')) {
                const outputText = Buffer.from(item.data).toString('utf8');
                if (/\b(Traceback|Error|Exception|Fail(ed|ure))\b/i.test(outputText)) {
                    return outputText.trim();
                }
            }
        }
    }
    // console.log(`No traceback error detected in cell ${cell.index}`);
    return undefined;
}

/**
 * Finds the most recently executed code cell in a notebook.
 */
export function findLastExecutedCell(notebook: NotebookDocument): NotebookCell | undefined {
     // Get cells, filter for Code cells with outputs and an execution order, then find the one with the highest order
     const executedCodeCells = notebook.getCells()
         .filter(cell =>
             cell.kind === NotebookCellKind.Code &&
             cell.outputs.length > 0 &&
             cell.executionSummary?.executionOrder !== undefined
         );

     if (executedCodeCells.length === 0) {
         console.log('No executed code cells found in the notebook.');
         return undefined;
     }

     // Sort by execution order descending and take the first one
     executedCodeCells.sort((a, b) => (b.executionSummary!.executionOrder!) - (a.executionSummary!.executionOrder!));

     const lastCell = executedCodeCells[0];
     console.log(`Found last executed cell: Index ${lastCell.index}, Order ${lastCell.executionSummary?.executionOrder}`);
     return lastCell;
 }


/**
 * Starts a simple loading animation by inserting and updating a Markdown cell.
 * Returns a function to stop the animation and remove the cell.
 */
export async function startNotebookLoadingAnimation(notebookEditor: NotebookEditor): Promise<() => Promise<void>> {
    const notebook = notebookEditor.notebook;
    // Insert at the end, or maybe after the selected/active cell? End is simpler.
    const insertIndex = notebook.cellCount;
    const loadingTextPrefix = '```\n🧠 Thinking...\n```'; // Use markdown code block for visibility
    const loadingCellData = new NotebookCellData(NotebookCellKind.Markup, loadingTextPrefix, 'markdown');

    const insertEdit = new WorkspaceEdit();
    const insertRange = new NotebookRange(insertIndex, insertIndex);
    insertEdit.set(notebook.uri, [NotebookEdit.insertCells(insertRange.start, [loadingCellData])]);

    try {
         await vscode.workspace.applyEdit(insertEdit);
    } catch (error) {
         console.error("Failed to insert loading animation cell:", error);
         // Return a no-op cleanup function if insertion failed
         return async () => {};
     }


    let dots = 0;
    let animationCellIndex = insertIndex; // The index where the cell was actually inserted

    const interval = setInterval(async () => {
        // Check if the cell still exists at the expected index
        try {
            const currentCell = notebook.cellAt(animationCellIndex);
             if (currentCell.kind !== NotebookCellKind.Markup) {
                 // Cell might have been changed or deleted, stop animation
                 clearInterval(interval);
                 return;
             }

            dots = (dots + 1) % 4;
            const newLoadingText = loadingTextPrefix.replace('...', '.'.repeat(dots));

            const updateEdit = new WorkspaceEdit();
             // Replace the content of the existing cell
             updateEdit.set(notebook.uri, [
                 NotebookEdit.replaceCells(new NotebookRange(animationCellIndex, animationCellIndex + 1), [
                     new NotebookCellData(NotebookCellKind.Markup, newLoadingText, 'markdown')
                 ])
             ]);
            await vscode.workspace.applyEdit(updateEdit);

        } catch (error) {
            // Error likely means the cell index is out of bounds (cell deleted)
             console.log("Loading animation cell likely deleted, stopping animation.");
            clearInterval(interval);
        }
    }, 750); // Slower update interval for notebooks maybe?

    // Return the cleanup function
    return async () => {
        clearInterval(interval);
         try {
             // Check if cell still exists before trying to delete
             notebook.cellAt(animationCellIndex); // This will throw if index is invalid

             const removeEdit = new WorkspaceEdit();
             removeEdit.set(notebook.uri, [NotebookEdit.deleteCells(new NotebookRange(animationCellIndex, animationCellIndex + 1))]);
             await vscode.workspace.applyEdit(removeEdit);
         } catch (error) {
             // Cell likely already removed or index invalid, log and ignore
             console.log("Could not remove animation cell (may have already been deleted):", error);
         }
    };
}

/**
 * Splits a response string into text and code parts.
 */
export function splitResponseIntoParts(response: string): ResponsePart[] {
    const parts: ResponsePart[] = [];
    // Improved regex to handle optional language and capture content robustly
    const codeBlockRegex = /```(\w+)?\s*([\s\S]*?)```/g;
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(response)) !== null) {
        // Add text part before the code block if it exists
        if (match.index > lastIndex) {
            const textContent = response.slice(lastIndex, match.index).trim();
            if (textContent) {
                parts.push({ type: 'text', content: textContent });
            }
        }

        // Add the code block part
        const language = match[1] || 'plaintext'; // Default to plaintext or python?
        const codeContent = match[2].trim();
        parts.push({ type: 'code', content: codeContent, language: language });

        lastIndex = match.index + match[0].length;
    }

    // Add any remaining text part after the last code block
    if (lastIndex < response.length) {
        const textContent = response.slice(lastIndex).trim();
        if (textContent) {
            parts.push({ type: 'text', content: textContent });
        }
    }

    return parts;
}


/**
 * Appends the response to the notebook, splitting it into Markdown and Code cells.
 */
export async function appendResponseToNotebook(notebookEditor: NotebookEditor, response: string): Promise<void> {
    const notebook = notebookEditor.notebook;
    // Insert after the last cell, or after the currently focused cell? End is safer.
    const insertIndex = notebook.cellCount;
    const parts = splitResponseIntoParts(response);

    if (parts.length === 0) {
        console.log("No parts to append to notebook.");
        return;
    }

    const cellsToAdd: NotebookCellData[] = parts.map(part => {
        if (part.type === 'code') {
            return new NotebookCellData(NotebookCellKind.Code, part.content, part.language || 'python'); // Default language
        } else {
            return new NotebookCellData(NotebookCellKind.Markup, part.content, 'markdown');
        }
    });

    const edit = new WorkspaceEdit();
    const range = new NotebookRange(insertIndex, insertIndex);
    edit.set(notebook.uri, [NotebookEdit.insertCells(range.start, cellsToAdd)]);

    try {
        const success = await vscode.workspace.applyEdit(edit);
        if (success) {
            console.log('Response parts appended to Jupyter Notebook');
            // Optional: Focus the first newly added cell?
            // vscode.commands.executeCommand('notebook.focusTop'); // Or focus last cell?
        } else {
            console.error('Failed to apply edit to append response to notebook');
            vscode.window.showErrorMessage('Failed to add response to notebook.');
        }
    } catch (error) {
        console.error('Error applying edit to append response to notebook:', error);
        vscode.window.showErrorMessage('Error adding response to notebook.');
    }
}

/**
 * Gets the text content of the selected code cells in the notebook editor.
 */
export function getSelectedCodeCellsContent(notebookEditor: NotebookEditor): string | undefined {
    const selectedCells = notebookEditor.selections
        .flatMap(selection => notebookEditor.notebook.getCells(selection)) // Get all cells in selections
        .filter(cell => cell.kind === NotebookCellKind.Code); // Filter for code cells

    if (selectedCells.length === 0) {
         // If no cells selected, maybe use the *active* cell if it's code?
         const activeCell = notebookEditor.selection?.start !== undefined ? notebookEditor.notebook.cellAt(notebookEditor.selection.start) : undefined;
          if (activeCell && activeCell.kind === NotebookCellKind.Code) {
              return activeCell.document.getText();
          }
        vscode.window.showInformationMessage('No code cells selected. Please select one or more code cells, or place cursor in one.');
        return undefined;
    }

    // Join content of selected code cells
    return selectedCells.map(cell => cell.document.getText()).join('\n\n---\n\n'); // Separator between cells
}

/**
 * Inserts a new code cell into the notebook.
 */
export async function insertCodeBlockToNotebook(notebookEditor: NotebookEditor, codeBlock: CodeBlock): Promise<void> {
    const notebook = notebookEditor.notebook;
    const insertIndex = notebook.cellCount; // Append to end
    const codeCell = new NotebookCellData(
        NotebookCellKind.Code,
        codeBlock.code,
        codeBlock.language || 'python' // Default language
    );

    const edit = new WorkspaceEdit();
    const range = new NotebookRange(insertIndex, insertIndex);
    edit.set(notebook.uri, [NotebookEdit.insertCells(range.start, [codeCell])]);

    try {
        await vscode.workspace.applyEdit(edit);
        console.log(`Inserted code block (${codeBlock.language || 'default'}) into notebook.`);
        // Optionally focus the new cell
    } catch (error) {
        console.error("Failed to insert code block into notebook:", error);
        vscode.window.showErrorMessage(`Failed to insert code block: ${error}`);
    }
}