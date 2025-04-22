// src/notebook/notebookService.ts
import * as vscode from 'vscode';
import { NotebookDocument, NotebookCell, NotebookEditor, NotebookDocumentChangeEvent } from 'vscode';
import { sendMessage } from '../chat/messageSender'; // Adjusted import
import {
    detectTracebackError,
    findLastExecutedCell,
    startNotebookLoadingAnimation,
    appendResponseToNotebook,
    getSelectedCodeCellsContent
} from './notebookUtils';
import { extractCodeFromNotebookResponse } from '../features/codeExtraction'; // Adjusted import

let isProcessingNotebookRequest = false; // Prevent concurrent requests

/**
 * Handles the main chat interaction initiated from a notebook context.
 * Uses selected cells or the active cell as context.
 */
export async function handleChatNotebook() {
    if (isProcessingNotebookRequest) {
        vscode.window.showInformationMessage('A notebook request is already being processed.');
        return;
    }

    const notebookEditor = vscode.window.activeNotebookEditor;
    if (!notebookEditor) {
        vscode.window.showInformationMessage('No active Jupyter Notebook editor found.');
        return;
    }

     isProcessingNotebookRequest = true;
     let stopLoadingAnimation: (() => Promise<void>) | undefined;


    try {
        const promptContent = getSelectedCodeCellsContent(notebookEditor);
        if (promptContent === undefined) {
             // Message already shown by getSelectedCodeCellsContent
            return;
        }

         // Start loading animation before sending the message
         stopLoadingAnimation = await startNotebookLoadingAnimation(notebookEditor);

        // Construct the prompt for the LLM
        // You might want to add more context, like file name or other cells if needed
        const fullPrompt = `Analyze or explain the following code from a Jupyter Notebook cell:\n\n\`\`\`python\n${promptContent}\n\`\`\`\n`; // Assuming python, adjust if needed

        const response = await sendMessage(fullPrompt); // Send to LLM
         console.log('Received response for notebook chat.');

         // Stop the animation *before* appending the response
         if (stopLoadingAnimation) {await stopLoadingAnimation();}
         stopLoadingAnimation = undefined; // Ensure it's not called again in finally

        await appendResponseToNotebook(notebookEditor, response); // Append formatted response

    } catch (error: any) {
        console.error("Error in handleChatNotebook:", error);
         if (stopLoadingAnimation) {
            await stopLoadingAnimation(); // Ensure animation stops on error
            stopLoadingAnimation = undefined;
        }
        vscode.window.showErrorMessage(`Error processing notebook chat: ${error.message}`);
    } finally {
         isProcessingNotebookRequest = false;
     }
}

/**
 * Handles the automatic detection and interaction for traceback errors.
 */
export async function handleTracebackError(notebook?: NotebookDocument, errorCell?: NotebookCell, errorText?: string) {
     if (isProcessingNotebookRequest) {
         console.log('Skipping traceback handling as another notebook request is active.');
         return;
     }

    const notebookEditor = vscode.window.activeNotebookEditor;
    if (!notebookEditor) {return;} // Needs an active editor

    // Use provided details or try to detect automatically
    notebook = notebook || notebookEditor.notebook;
    if (!errorCell || !errorText) {
         errorCell = findLastExecutedCell(notebook);
         if (!errorCell) {return;} // No cell found
         errorText = detectTracebackError(errorCell);
         if (!errorText) {return;} // No error detected in the cell
     }

    const userChoice = await vscode.window.showQuickPick(['Yes', 'No'], {
        placeHolder: 'Error detected in notebook cell. Get help explaining and fixing it?',
        ignoreFocusOut: true // Keep prompt open
    });

    if (userChoice !== 'Yes') {
        return; // User declined help
    }

     isProcessingNotebookRequest = true;
     let stopLoadingAnimation: (() => Promise<void>) | undefined;

    try {
        const codeContent = errorCell.document.getText();
        const prompt = `I encountered an error in my Jupyter Notebook.\n\nCell Code:\n\`\`\`python\n${codeContent}\n\`\`\`\n\nError Traceback:\n\`\`\`\n${errorText}\n\`\`\`\n\nCan you explain the cause of this error and suggest how to fix the code?`;

         stopLoadingAnimation = await startNotebookLoadingAnimation(notebookEditor);
        const response = await sendMessage(prompt);
         console.log('Received response for notebook error.');

         if (stopLoadingAnimation) {await stopLoadingAnimation();}
         stopLoadingAnimation = undefined;

        await appendResponseToNotebook(notebookEditor, response);

        // Maybe offer to extract code from the response automatically?
         const hasCode = /```(\w+)?\s*([\s\S]*?)```/.test(response);
          if (hasCode) {
              const extractChoice = await vscode.window.showQuickPick(['Yes', 'No'], {
                  placeHolder: 'The response contains code suggestions. Extract them into new cells?'
              });
              if (extractChoice === 'Yes') {
                  await extractCodeFromNotebookResponse(notebookEditor); // Call extraction feature
              }
          }

    } catch (error: any) {
        console.error("Error handling traceback:", error);
        if (stopLoadingAnimation) {await stopLoadingAnimation();}
        stopLoadingAnimation = undefined;
        vscode.window.showErrorMessage(`Error getting help for traceback: ${error.message}`);
    } finally {
         isProcessingNotebookRequest = false;
    }
}

/**
 * Listener for notebook document changes, specifically looking for cell executions
 * that might result in errors.
 */
export async function handleNotebookDocumentChange(e: NotebookDocumentChangeEvent) {
    // Check if the change involves cell execution summary updates
     const executionChanges = e.cellChanges.filter(change => change.executionSummary !== undefined);
     // Also check cell output changes, as errors might appear there without execution summary changing immediately
     const outputChanges = e.cellChanges.filter(change => change.outputs !== undefined && change.outputs.length > (change.cell.outputs?.length ?? 0));


    if (executionChanges.length > 0 || outputChanges.length > 0) {
        const notebook = e.notebook;
        // Find the cell that most likely just finished executing and might have an error
        // This could be the last executed cell, or one of the changed cells
        const lastExecuted = findLastExecutedCell(notebook); // This is likely the most reliable
        if (lastExecuted) {
             const error = detectTracebackError(lastExecuted);
             if (error) {
                 console.log(`Error detected in cell ${lastExecuted.index} after execution/output change.`);
                 // Automatically trigger error handling without prompt for now
                 // Could add a setting to control automatic triggering vs. manual command later
                 await handleTracebackError(notebook, lastExecuted, error);
             }
         }
    }
}