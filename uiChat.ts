// uiChat.ts

import * as vscode from 'vscode';
import { getCurrentModel, sendMessage, getActivePrompt } from './chatUtils';
import { getWorkspaceFiles, getFileContent } from './workspaceUtils';
import { getFuzzyFileList } from './workspaceUtils';


export class UiChatViewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private _messageHistory: { message: string; isUser: boolean }[] = [];

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async data => {
            switch (data.type) {
                case 'sendMessage':
                    await this.handleChatMessage(data.message);
                    break;
                case 'getAutocompleteSuggestions':
                    await this.handleAutocompleteSuggestions(data.partial);
                    break;
                case 'clearChat':
                    this.clearChat();
                    break;
            }
        });

        // Restore message history when the view becomes visible
        webviewView.onDidChangeVisibility(() => {
            if (webviewView.visible && this._messageHistory.length > 0) {
                console.log('Restoring message history');
                this._view?.webview.postMessage({ type: 'restoreHistory', messages: this._messageHistory });
            }
        });
    }


    private async handleAutocompleteSuggestions(partial: string) {
        console.log('Handling autocomplete suggestions for:', partial);
        const suggestions = await getFuzzyFileList(partial);
        console.log('Autocomplete suggestions:', suggestions);
        this._view?.webview.postMessage({ type: 'autocompleteSuggestions', suggestions });
    }

    private async handleChatMessage(message: string) {
        if (!this._view) {
            console.log('View is not available');
            return;
        }

        console.log('Handling chat message:', message);
        this._messageHistory.push({ message, isUser: true });
        this._view.webview.postMessage({ type: 'addMessage', message, isUser: true });

        const activePrompt = getActivePrompt();
        console.log('Active prompt:', activePrompt);

        const fileReferences = message.match(/@[\w-/.]+/g) || [];
        let workspaceContext = '';

        for (const reference of fileReferences) {
            const filePath = reference.slice(1);  // Remove the '@' symbol
            try {
                const content = await getFileContent(filePath);
                workspaceContext += `File: ${filePath}\n${content}\n\n`;
            } catch (error) {
                console.error(`Error loading file ${filePath}:`, error);
                workspaceContext += `Error loading file ${filePath}\n`;
            }
        }

        const fullPrompt = `$${workspaceContext ? `Provided Files:\n${workspaceContext}\n` : ''}User: ${message}\n`;

        console.log('Full prompt created');

        try {
            console.log('Sending message to model');
            const response = await sendMessage(fullPrompt);
            console.log('Received response from model:', response);
            this._messageHistory.push({ message: response, isUser: false });
            this._view.webview.postMessage({ type: 'addMessage', message: response, isUser: false });
        } catch (error) {
            console.error('Error in handleChatMessage:', error);
            const errorMessage = `Error: ${error}`;
            this._messageHistory.push({ message: errorMessage, isUser: false });
            this._view.webview.postMessage({ type: 'addMessage', message: errorMessage, isUser: false });
            vscode.window.showErrorMessage(errorMessage);
        }
    }

    private clearChat() {
        this._messageHistory = [];
        if (this._view) {
            this._view.webview.postMessage({ type: 'clearChat' });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'style.css'));
    
        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link href="${styleUri}" rel="stylesheet">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-core.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/plugins/autoloader/prism-autoloader.min.js"></script>
                <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism-tomorrow.min.css" rel="stylesheet" />
                <title>Promptly Chat</title>
            </head>
            <body>
                <div id="chat-container"></div>
                <div id="input-container">
                    <div id="autocomplete-container"></div>
                    <textarea id="message-input" placeholder="Type your message... Use @ to reference files"></textarea>
                    <div id="button-container">
                        <button id="send-button" class="button">Send</button>
                        <button id="menu-button" class="button">⋮</button>
                        <div id="popup-menu" class="popup-menu">
                            <button id="clear-button" class="menu-item">Clear Chat</button>
                        </div>
                    </div>
                </div>
                <script src="${scriptUri}"></script>
            </body>
            </html>`;
    }
}