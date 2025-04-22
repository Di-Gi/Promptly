// src/ui/uiChat.ts
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

// Adjust import paths based on the new structure
import { sendMessage } from '../chat/messageSender';
import { getActivePrompt } from '../config/promptService';
// Assuming workspaceUtils is now in common/
import { getFileContent, getFuzzyFileList } from '../common/workspaceUtils';
// getCurrentModel might not be needed directly here unless displaying it in the UI
// import { getCurrentModel } from '../config/modelService';

export class UiChatViewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    // Simple history, consider a more robust solution for large histories
     private _messageHistory: { message: string; isUser: boolean; files?: string[] }[] = [];
     private _isViewVisible = false; // Track visibility

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;
        this._isViewVisible = true; // Initially visible

        webviewView.webview.options = {
            enableScripts: true,
            // IMPORTANT: Restrict resource loading to the media directory
            localResourceRoots: [vscode.Uri.joinPath(this._extensionUri, 'media')]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async data => {
            console.log("[Extension] Webview message received:", data.type); // Log backend received messages
            switch (data.type) {
                case 'sendMessage':
                    // Add user message immediately to history and UI
                    const userMessage = { message: data.message, isUser: true, files: data.attachedFiles || [] };
                     this._messageHistory.push(userMessage);
                    // Send raw message to UI to preserve links etc., escaping happens client-side if needed there
                     this._view?.webview.postMessage({ type: 'addMessageRaw', html: this.formatUserMessageForDisplay(userMessage), isUser: true}); // Reflect immediately
                     // Process the message
                    await this.handleChatMessage(data.message, data.attachedFiles || []);
                    break;
                case 'getAutocompleteSuggestions':
                    await this.handleAutocompleteSuggestions(data.partial);
                    break;
                case 'openFilePreviewRequest': // Renamed for clarity
                     await this.handleFilePreview(data.filePath);
                     break;
                case 'openFile':
                     await this.handleOpenFile(data.file);
                     break;
                case 'clearChat':
                     this.clearChat();
                     break;
                 case 'webviewReady': // Message from webview when it's ready
                     console.log("[Extension] Webview reported ready. Restoring history if needed.");
                     this.restoreHistory();
                     break;
                 // Add other message types as needed
                 case 'log': // Allow webview to send logs
                     console.log(`[Webview Log] ${data.message}`);
                     break;
                case 'showError': // Allow webview to show error messages via VS Code API
                     vscode.window.showErrorMessage(data.message);
                     break;
            }
        });

        webviewView.onDidChangeVisibility(() => {
            this._isViewVisible = webviewView.visible;
            if (this._isViewVisible) {
                 console.log("[Extension] Webview became visible.");
                 // History restoration is handled by 'webviewReady' or explicit request
                 this.restoreHistory();
            } else {
                console.log("[Extension] Webview became hidden.");
            }
        });

         webviewView.onDidDispose(() => {
             console.log("[Extension] Webview disposed.");
             this._view = undefined;
             this._isViewVisible = false;
             // Optionally clear history on dispose? Or keep it for next session?
             this._messageHistory = []; // Clear history when view is disposed
         });
    }

    // Format user message for display, handling attached files
    private formatUserMessageForDisplay(userMessage: { message: string; isUser: boolean; files?: string[] }): string {
        // Basic HTML escaping for the message text itself
        const escapeHtml = (unsafe: string) =>
            unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");

        let displayMessage = escapeHtml(userMessage.message);

        if (userMessage.files && userMessage.files.length > 0) {
            displayMessage += `<br><div class="attached-files-display"><strong>Attached:</strong> ` +
                userMessage.files
                    .map(f => {
                        const escapedFile = escapeHtml(f);
                        const fileName = escapeHtml(path.basename(f));
                        return `<span class="attached-file-tag" title="Click to open ${escapedFile}"><a href="#" class="attached-file-link" data-file="${escapedFile}">${fileName}</a></span>`;
                    })
                    .join(' ');
            displayMessage += `</div>`;
        }

        return displayMessage;
    }


    // Method to restore history, called on visibility change or webview ready
     private restoreHistory() {
         if (this._view && this._isViewVisible && this._messageHistory.length > 0) {
             console.log(`[Extension] Restoring ${this._messageHistory.length} messages to webview.`);
             // Send the history, letting the webview handle rendering (addMessageRaw/addMessage)
             this._view.webview.postMessage({ type: 'restoreHistory', messages: this._messageHistory });
         }
     }

    private async handleAutocompleteSuggestions(partial: string) {
        if (!partial || partial.length < 1) { // Basic validation
            this._view?.webview.postMessage({ type: 'autocompleteSuggestions', suggestions: [] });
            return;
        }
        console.log('[Extension] Handling autocomplete suggestions for:', partial);
        try {
             const suggestions = await getFuzzyFileList(partial);
             console.log('[Extension] Autocomplete suggestions found:', suggestions.length);
             this._view?.webview.postMessage({ type: 'autocompleteSuggestions', suggestions });
         } catch (error: any) {
             console.error("[Extension] Error getting autocomplete suggestions:", error);
              this._view?.webview.postMessage({ type: 'autocompleteSuggestions', suggestions: [] }); // Send empty on error
              // Optionally notify user
              // vscode.window.showWarningMessage(`Could not get file suggestions: ${error.message}`);
          }
    }

    private async handleFilePreview(filePath: string) {
        console.log('[Extension] Handling file preview request for:', filePath);
        if (!this._view) {return;}
        try {
            const content = await getFileContent(filePath);
            // Limit preview content size?
            const maxPreviewSize = 5000; // characters
            const truncatedContent = content.length > maxPreviewSize
                ? content.substring(0, maxPreviewSize) + "\n\n... (file truncated for preview) ..."
                : content;
            this._view.webview.postMessage({ type: 'showFilePreview', filePath, content: truncatedContent }); // Use 'showFilePreview'
        } catch (error: any) {
            console.error(`[Extension] Error loading file preview for ${filePath}:`, error);
            // Send error back to webview to display?
            this._view.webview.postMessage({ type: 'filePreviewError', filePath, message: `Could not load preview: ${error.message}` });
            // vscode.window.showErrorMessage(`Error loading file preview: ${error.message}`);
        }
    }

    private async handleOpenFile(relativeFilePath: string) {
        console.log('[Extension] Handling open file request for:', relativeFilePath);
        if (!vscode.workspace.workspaceFolders) {
            vscode.window.showErrorMessage('No workspace folder open.');
            return;
        }

        // Try to find the file in workspace folders
        let fileUri: vscode.Uri | undefined;
        for (const folder of vscode.workspace.workspaceFolders) {
            const potentialUri = vscode.Uri.joinPath(folder.uri, relativeFilePath);
            try {
                await vscode.workspace.fs.stat(potentialUri); // Check if file exists
                fileUri = potentialUri;
                break;
            } catch {
                // File not in this folder, try next
            }
        }

        if (fileUri) {
             try {
                 const doc = await vscode.workspace.openTextDocument(fileUri);
                 await vscode.window.showTextDocument(doc);
                 console.log(`[Extension] Opened file: ${fileUri.fsPath}`);
             } catch (error: any) {
                 console.error(`[Extension] Error opening file ${relativeFilePath}:`, error);
                 vscode.window.showErrorMessage(`Failed to open file: ${error.message}`);
             }
        } else {
             vscode.window.showErrorMessage(`File not found in workspace: ${relativeFilePath}`);
         }
    }

    private async handleChatMessage(message: string, attachedFiles: string[]) {
        if (!this._view) {
            console.error('[Extension] View is not available for sending message.');
            return;
        }

         console.log('[Extension] Processing chat message for LLM. Message:', message, 'Attached files:', attachedFiles);
         this._view.webview.postMessage({ type: 'showLoading' }); // Tell UI to show loading indicator

        // Get active prompt and construct context
        const activePrompt = getActivePrompt(); // System prompt
        let context = '';
        const referencedFiles = new Set<string>(attachedFiles); // Use Set for efficient lookup

        // Process @-mentions in the message *server-side* as well to ensure context is correct
        const fileMentions = message.match(/@([\w\-./\\]+)/g) || []; // Match paths with ., -, /
        for (const mention of fileMentions) {
            const filePath = mention.slice(1); // Remove '@'
            referencedFiles.add(filePath);
        }

        // Fetch content for all referenced files (attached + @mentioned)
        if (referencedFiles.size > 0) {
             context += "Referenced Files:\n";
             for (const filePath of referencedFiles) {
                 try {
                     const content = await getFileContent(filePath);
                     // Add token limiting here if needed
                     const maxFileTokens = 2000; // Example limit
                     const limitedContent = content.length > maxFileTokens
                         ? content.substring(0, maxFileTokens) + "\n... (content truncated) ..."
                         : content;
                     context += `--- File: ${filePath} ---\n${limitedContent}\n\n`;
                 } catch (error) {
                     console.error(`[Extension] Error loading referenced file ${filePath}:`, error);
                     context += `--- File: ${filePath} [Error loading file content] ---\n\n`;
                 }
             }
         }

        // Construct the final prompt for the LLM
        // Prepend context to the user's message
        const fullPrompt = `${context}${message}`; // Let sendMessage handle system prompt internally

         console.log(`[Extension] Sending prompt to LLM (Context: ${context.length} chars, Message: ${message.length} chars)...`);

        try {
            const response = await sendMessage(fullPrompt); // Send combined context + message
             console.log('[Extension] LLM response received.');

             const aiMessage = { message: response, isUser: false };
             this._messageHistory.push(aiMessage); // Add AI response to history
             this._view?.webview.postMessage({ type: 'addMessage', message: aiMessage.message, isUser: false }); // Send to UI

        } catch (error: any) {
            console.error('[Extension] Error during LLM communication in UI Chat:', error);
            const errorMessage = `Sorry, I encountered an error: ${error.message}`;
             const errorResponseMessage = { message: errorMessage, isUser: false };
             this._messageHistory.push(errorResponseMessage); // Add error to history
            this._view?.webview.postMessage({ type: 'addMessage', message: errorMessage, isUser: false }); // Show error in UI
            // No need for vscode.window.showErrorMessage here, error shown in chat
        } finally {
             this._view?.webview.postMessage({ type: 'hideLoading' }); // Hide loading indicator
         }
    }

    private clearChat() {
        console.log("[Extension] Clearing chat history.");
        this._messageHistory = [];
        if (this._view) {
            // Tell webview to clear its display
            this._view.webview.postMessage({ type: 'clearChat' });
        }
    }

    // Updated HTML Generation
    private _getHtmlForWebview(webview: vscode.Webview): string {
         // Use helper to get webview URIs for local resources
         const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js'));
         const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'style.css'));
          // Get URI for Monaco loader (consider packaging it locally)
         const monacoLoaderUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'node_modules', 'monaco-editor', 'min', 'vs', 'loader.js'));
         const monacoBaseUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'node_modules', 'monaco-editor', 'min', 'vs'));

         // Nonce for inline scripts/styles if needed (or use hashes)
         const nonce = getNonce();

         // **Removed** hardcoded preview elements from #file-preview-container
         // **Added** class="button" to send/menu buttons
         // **Added** nonce to scripts

         return `<!DOCTYPE html>
             <html lang="en">
             <head>
                 <meta charset="UTF-8">
                 <!-- CSP enforcing loading from extension media and Monaco CDN (or local path) -->
                 <meta http-equiv="Content-Security-Policy" content="
                     default-src 'none';
                     style-src ${webview.cspSource} 'unsafe-inline';
                     script-src 'nonce-${nonce}' ${webview.cspSource};
                     font-src ${webview.cspSource};
                     img-src ${webview.cspSource} https: data:;
                     connect-src https:;
                     worker-src blob: ${webview.cspSource};
                 ">
                 <meta name="viewport" content="width=device-width, initial-scale=1.0">
                 <link href="${styleUri}" rel="stylesheet">
                 <!-- Use local Monaco loader URI if available -->
                 <!-- <script nonce="${nonce}" src="${monacoLoaderUri}"></script> -->
                 <title>Promptly Chat</title>
             </head>
             <body>
                <div id="chat-container">
                    <!-- Messages dynamically added -->
                </div>

                <!-- File preview container - initially empty and hidden by CSS -->
                <div id="file-preview-container">
                    <!-- Preview elements added dynamically by JS -->
                </div>

                <div id="input-area">
                    <!-- Autocomplete list positioned absolutely -->
                    <div id="autocomplete-list"></div>

                    <textarea id="message-input" placeholder="Type your message... Use @ for files." rows="1"></textarea>

                    <div id="input-controls">
                        <button id="send-button" class="button" title="Send Message">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z"/></svg>
                        </button>
                        <button id="menu-button" class="button" title="Chat Options">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="16" height="16"><path d="M8 4a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3ZM8 8.5a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3ZM9.5 11.5a1.5 1.5 0 1 0-3 0 1.5 1.5 0 0 0 3 0Z"/></svg>
                        </button>
                        <div id="popup-menu" class="popup-menu">
                            <button id="clear-button" class="menu-item">Clear Chat</button>
                            <!-- Add more menu items if needed -->
                        </div>
                    </div>
                </div>

                 <div id="loading-indicator" class="loading-container" style="display: none;">
                    <div class="bouncing-bar">
                        <div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>
                    </div>
                 </div>

                  <!-- Prism JS for syntax highlighting (consider local copy) -->
                  <!-- Check CSP if using external source -->
                 <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
                 <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
                 <!-- Main webview script -->
                 <script nonce="${nonce}" src="${scriptUri}"></script>
             </body>
             </html>`;
    }
}

// Simple nonce generator
function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}