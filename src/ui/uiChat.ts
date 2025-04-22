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
            localResourceRoots: [vscode.Uri.joinPath(this._extensionUri, 'media')] // Restrict root
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async data => {
            console.log("Webview message received:", data.type);
            switch (data.type) {
                case 'sendMessage':
                    // Add user message immediately to history and UI
                    const userMessage = { message: data.message, isUser: true, files: data.attachedFiles || [] };
                     this._messageHistory.push(userMessage);
                     this._view?.webview.postMessage({ type: 'addMessage', message: userMessage.message, isUser: true, files: userMessage.files }); // Reflect immediately
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
                     console.log("Webview reported ready. Restoring history if needed.");
                     this.restoreHistory();
                     break;
            }
        });

        webviewView.onDidChangeVisibility(() => {
            this._isViewVisible = webviewView.visible;
            if (this._isViewVisible) {
                 console.log("Webview became visible.");
                 // History restoration is handled by 'webviewReady' or explicit request
                 this.restoreHistory();
            } else {
                console.log("Webview became hidden.");
            }
        });

         webviewView.onDidDispose(() => {
             console.log("Webview disposed.");
             this._view = undefined;
             this._isViewVisible = false;
             // Optionally clear history on dispose? Or keep it for next session?
             // this._messageHistory = [];
         });
    }

    // Method to restore history, called on visibility change or webview ready
     private restoreHistory() {
         if (this._view && this._isViewVisible && this._messageHistory.length > 0) {
             console.log(`Restoring ${this._messageHistory.length} messages to webview.`);
             this._view.webview.postMessage({ type: 'restoreHistory', messages: this._messageHistory });
         }
     }

    private async handleAutocompleteSuggestions(partial: string) {
        if (!partial || partial.length < 1) { // Basic validation
            this._view?.webview.postMessage({ type: 'autocompleteSuggestions', suggestions: [] });
            return;
        }
        // console.log('Handling autocomplete suggestions for:', partial);
        try {
             const suggestions = await getFuzzyFileList(partial);
             // console.log('Autocomplete suggestions:', suggestions);
             this._view?.webview.postMessage({ type: 'autocompleteSuggestions', suggestions });
         } catch (error) {
             console.error("Error getting autocomplete suggestions:", error);
              this._view?.webview.postMessage({ type: 'autocompleteSuggestions', suggestions: [] }); // Send empty on error
          }
    }

    private async handleFilePreview(filePath: string) {
        console.log('Handling file preview request for:', filePath);
        if (!this._view) {return;}
        try {
            const content = await getFileContent(filePath);
            this._view.webview.postMessage({ type: 'showFilePreview', filePath, content }); // Use 'showFilePreview'
        } catch (error: any) {
            console.error(`Error loading file preview for ${filePath}:`, error);
            vscode.window.showErrorMessage(`Error loading file preview: ${error.message}`);
             // Optionally inform webview about the error
             // this._view.webview.postMessage({ type: 'filePreviewError', filePath, message: error.message });
        }
    }

    private async handleOpenFile(relativeFilePath: string) {
        console.log('Handling open file request for:', relativeFilePath);
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
                 console.log(`Opened file: ${fileUri.fsPath}`);
             } catch (error: any) {
                 console.error(`Error opening file ${relativeFilePath}:`, error);
                 vscode.window.showErrorMessage(`Failed to open file: ${error.message}`);
             }
        } else {
             vscode.window.showErrorMessage(`File not found in workspace: ${relativeFilePath}`);
         }
    }

    private async handleChatMessage(message: string, attachedFiles: string[]) {
        if (!this._view) {
            console.error('View is not available for sending message.');
            return;
        }

         console.log('Processing chat message for LLM. Message:', message, 'Attached files:', attachedFiles);
         this._view.webview.postMessage({ type: 'showLoading' }); // Tell UI to show loading indicator


        // Get active prompt and construct context
        const activePrompt = getActivePrompt(); // System prompt
        let context = '';
        const referencedFiles = new Set<string>(attachedFiles); // Use Set for efficient lookup

        // Process @-mentions in the message
        const fileMentions = message.match(/@([\w\-./\\]+)/g) || []; // Match paths with ., -, /
        for (const mention of fileMentions) {
            const filePath = mention.slice(1); // Remove '@'
            referencedFiles.add(filePath);
        }

        // Fetch content for all referenced files
        if (referencedFiles.size > 0) {
             context += "Referenced Files:\n";
             for (const filePath of referencedFiles) {
                 try {
                     const content = await getFileContent(filePath);
                     // Truncate long file content? Add token counting?
                     context += `--- File: ${filePath} ---\n${content}\n\n`;
                 } catch (error) {
                     console.error(`Error loading referenced file ${filePath}:`, error);
                     context += `--- File: ${filePath} [Error loading content] ---\n\n`;
                 }
             }
         }

        // Construct the final prompt for the LLM
        const fullPrompt = `${context}${message}`; // Let sendMessage handle system prompt internally

         console.log(`Sending prompt to LLM (length ${fullPrompt.length})...`);

        try {
            const response = await sendMessage(fullPrompt); // Send combined context + message
             console.log('LLM response received.');

             const aiMessage = { message: response, isUser: false };
             this._messageHistory.push(aiMessage); // Add AI response to history
             this._view?.webview.postMessage({ type: 'addMessage', message: aiMessage.message, isUser: false }); // Send to UI

        } catch (error: any) {
            console.error('Error during LLM communication in UI Chat:', error);
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
        console.log("Clearing chat history.");
        this._messageHistory = [];
        if (this._view) {
            this._view.webview.postMessage({ type: 'clearChat' });
        }
    }

    // Make sure the HTML includes necessary elements and references the correct JS/CSS URIs
    private _getHtmlForWebview(webview: vscode.Webview): string {
         // Use helper to get webview URIs
         const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js'));
         const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'style.css'));

         // Use a nonce for Content Security Policy
         const nonce = getNonce(); // Implement getNonce() function

         return `<!DOCTYPE html>
             <html lang="en">
             <head>
                 <meta charset="UTF-8">
                 <!-- CSP -->
                 <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} https:;">
                 <meta name="viewport" content="width=device-width, initial-scale=1.0">
                 <link href="${styleUri}" rel="stylesheet">
                  <!-- Consider including library CSS directly or using webview URI -->
                 <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
                 <title>Promptly Chat</title>
             </head>
             <body>
                 <div id="chat-container">
                     <!-- Messages will be added here -->
                 </div>

                 <div id="file-preview-container" class="file-preview">
                     <button id="close-preview-button" class="close-button">&times;</button>
                     <h4 id="preview-file-path"></h4>
                     <pre><code id="preview-content" class="language-plaintext"></code></pre>
                 </div>

                 <div id="input-area">
                     <div id="autocomplete-list"></div>
                      <textarea id="message-input" placeholder="Type your message... Use @ for files." rows="3"></textarea>
                     <div id="input-controls">
                         <button id="send-button" title="Send Message">
                             <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z"/></svg>
                         </button>
                         <button id="menu-button" title="Chat Options">
                             <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path fill-rule="evenodd" d="M10.5 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm0 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm0 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0z" clip-rule="evenodd" /></svg>
                         </button>
                         <div id="popup-menu" class="popup-menu">
                             <button id="clear-button" class="menu-item">Clear Chat</button>
                             <!-- Add more menu items if needed -->
                         </div>
                     </div>
                 </div>
                 <div id="loading-indicator" style="display: none;">Thinking...</div>

                  <!-- Include Prism JS - consider hosting locally or using webview URI -->
                 <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
                 <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
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
