// src/chat/chatService.ts
import * as vscode from 'vscode';
import { sendMessage } from './messageSender';
import { getPrompt, getInsertPosition } from '../editor/editorService';
import { appendResponseToFile, startRequestAnimation } from '../editor/responseRenderer';
import { handleChatNotebook } from '../notebook/notebookService'; // Delegate notebook chats
import { AnimationControl } from '../common/types';

let isProcessingRequest = false; // Global lock for chat requests

/**
 * Main entry point for handling chat requests from the editor or commands.
 * Determines the context (editor/notebook) and delegates accordingly.
 * Manages the processing lock and basic error handling.
 */
export async function handleChat(promptOverride?: string) {
    if (isProcessingRequest) {
        vscode.window.showInformationMessage('A request is already being processed. Please wait.');
        return;
    }

    isProcessingRequest = true;
     let animationControl: AnimationControl | undefined;

    try {
        const editor = vscode.window.activeTextEditor;
        const notebookEditor = vscode.window.activeNotebookEditor;

        // --- Context Determination ---
        if (notebookEditor && (!editor || vscode.window.activeTextEditor !== editor)) {
             // If a notebook is active and *not* a text editor within it (like cell input focus)
            console.log("Handling chat in Notebook context.");
            // Notebook service handles its own UI (loading animation, appending)
            await handleChatNotebook(); // Prompt override not typically used here?
        } else if (editor) {
            // --- Text Editor Context ---
            console.log("Handling chat in Text Editor context.");
            const currentDocument = editor.document; // Capture document early
            const prompt = promptOverride ?? getPrompt(editor); // Use override or get from editor
            const insertPosition = getInsertPosition(editor);
            const fileType = currentDocument.languageId;

            if (!prompt && !promptOverride) {
                 vscode.window.showInformationMessage('Cannot send an empty prompt.');
                 return; // Exit early if prompt is empty
             }

            console.log(`Sending prompt (length ${prompt.length}) from ${fileType} file.`);

             animationControl = await startRequestAnimation(editor, insertPosition);

            const response = await sendMessage(prompt); // Get response from LLM
            console.log(`Received response (length ${response?.length ?? 0}).`);

            // Stop animation before rendering
            // await animationControl.stop(); // appendResponseToFile now handles this

            if (response) {
                 // Ensure we are still in the same editor/document before rendering
                 if (vscode.window.activeTextEditor === editor && editor.document === currentDocument) {
                    await appendResponseToFile(editor, response, animationControl);
                    animationControl = undefined; // Handled by append function now
                 } else {
                     console.warn("Editor or document changed before response could be rendered.");
                     vscode.window.showWarningMessage("Editor changed; response not inserted. You can find it in the logs if needed."); // Or copy to clipboard?
                     // Ensure animation cleanup if append didn't happen
                      if (animationControl) {await animationControl.stop();}
                  }
            } else {
                 // Handle cases where sendMessage resolved but returned nothing (should ideally throw error)
                 vscode.window.showErrorMessage('Received an empty response from the model.');
                 if (animationControl) {await animationControl.stop();}
             }

        } else {
            vscode.window.showInformationMessage('No active editor or notebook found to initiate chat.');
        }

    } catch (error: any) {
        console.error("Error during handleChat:", error);
        if (animationControl) {
             try { await animationControl.stop(); } catch (animError) { console.error("Error stopping animation during error handling:", animError); }
         }
        vscode.window.showErrorMessage(`Chat failed: ${error.message}`);
    } finally {
        isProcessingRequest = false;
    }
}