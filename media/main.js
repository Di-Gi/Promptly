// media/main.js

// Ensure Prism doesn't run automatically if loaded early
if (typeof Prism !== 'undefined') {
    Prism.manual = true;
}

(function() {
    // Check if running in VS Code webview context
    const isVsCode = typeof acquireVsCodeApi !== 'undefined';
    const vscode = isVsCode ? acquireVsCodeApi() : null;
    const postMessage = (message) => {
        if (vscode) {
            vscode.postMessage(message);
        } else {
            console.warn("VS Code API not available. Message not sent:", message);
        }
    };

    // --- DOM Elements ---
    const chatContainer = document.getElementById('chat-container');
    const filePreviewContainer = document.getElementById('file-preview-container');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const menuButton = document.getElementById('menu-button');
    const popupMenu = document.getElementById('popup-menu');
    const clearButton = document.getElementById('clear-button');
    const autocompleteList = document.getElementById('autocomplete-list'); // Corrected ID
    const inputArea = document.getElementById('input-area');
    const loadingIndicator = document.getElementById('loading-indicator');


    // --- State ---
    let selectedSuggestionIndex = -1;
    let currentSuggestions = [];
    let isAutocompleteActive = false;
    let attachedFiles = new Set(); // Use Set for uniqueness
    let currentMentionInfo = null; // Store info about the active @mention {start, end, partial}


    // --- Initial Setup & Event Listeners ---
    if (sendButton) {sendButton.addEventListener('click', handleSendMessage);}
    if (menuButton) {menuButton.addEventListener('click', togglePopupMenu);}
    if (clearButton) {clearButton.addEventListener('click', handleClearChat);}
    if (messageInput) {
        messageInput.addEventListener('input', handleInput);
        messageInput.addEventListener('keydown', handleKeydown);
        messageInput.addEventListener('focus', () => hideAutocompleteList()); // Hide on focus if user clicks away
        messageInput.addEventListener('blur', () => {
            // Delay hiding autocomplete to allow clicks on suggestions
            setTimeout(() => {
                if (!autocompleteList.matches(':hover')) { // Don't hide if mouse is over the list
                    hideAutocompleteList();
                }
            }, 150);
        });
    }

    // Close popup menu if clicked outside
    document.addEventListener('click', (event) => {
        if (popupMenu && menuButton && !menuButton.contains(event.target) && !popupMenu.contains(event.target)) {
            popupMenu.classList.remove('show');
        }
        // Close autocomplete if clicked outside input and list
        if (autocompleteList && messageInput && !messageInput.contains(event.target) && !autocompleteList.contains(event.target)) {
            hideAutocompleteList();
        }
    });

     // Handle clicks on dynamically added elements (copy, file links, remove preview)
     document.addEventListener('click', handleDynamicClicks);

    // Listen for messages from the VS Code extension
    window.addEventListener('message', handleExtensionMessage);

    // --- Functions ---

    function logToExtension(message) {
        postMessage({ type: 'log', message: `[Webview] ${message}` });
    }

    function togglePopupMenu(event) {
        event.stopPropagation();
        popupMenu.classList.toggle('show');
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {return '';}
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }


    function addMessage(htmlContent, isUser) {
        if (!chatContainer) {return;}
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', isUser ? 'user-message' : 'assistant-message');
        messageElement.innerHTML = htmlContent; // Assumes HTML is safe or already sanitized
        chatContainer.appendChild(messageElement);
        // Highlight code blocks within the new message
        if (typeof Prism !== 'undefined') {
             messageElement.querySelectorAll('pre code').forEach((block) => {
                 try {
                     Prism.highlightElement(block);
                 } catch (e) {
                     logToExtension(`Prism highlighting failed: ${e}`);
                 }
             });
         }
        scrollToBottom();
    }

    // Specific function to render assistant messages (handles markdown/code blocks)
    function renderAssistantMessage(message) {
        const codeBlockRegex = /```(\w*)\n?([\s\S]*?)```/g;
        let lastIndex = 0;
        let renderedHtml = '';

        // Basic markdown and code block parsing
        message.replace(codeBlockRegex, (match, lang, code, offset) => {
            // Render text before the code block
            renderedHtml += renderTextContent(message.slice(lastIndex, offset));
            // Render the code block
            renderedHtml += renderCodeBlock(lang, code);
            lastIndex = offset + match.length;
            return match; // Necessary for String.replace
        });
        // Render any remaining text after the last code block
        renderedHtml += renderTextContent(message.slice(lastIndex));

        addMessage(renderedHtml, false); // Add the fully rendered assistant message
    }

    function renderTextContent(text) {
        // Apply basic markdown: bold, italics, inline code, links, newlines
        return escapeHtml(text)
            .replace(/`([^`]+?)`/g, '<code>$1</code>') // Inline code
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.+?)\*/g, '<em>$1</em>')   // Italics
             // Simple URL detection (needs improvement for complex URLs)
            .replace(/(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/ig, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>')
            .replace(/\n/g, '<br>'); // Newlines
    }

    function renderCodeBlock(lang, code) {
         const safeLang = escapeHtml(lang || 'plaintext');
         // Trim leading/trailing whitespace/newlines from code before highlighting
         const trimmedCode = code.trim();
         let highlightedCode = escapeHtml(trimmedCode); // Fallback if Prism fails

        if (typeof Prism !== 'undefined' && Prism.languages[safeLang]) {
            try {
                highlightedCode = Prism.highlight(trimmedCode, Prism.languages[safeLang], safeLang);
            } catch (e) {
                logToExtension(`Prism highlighting error for lang "${safeLang}": ${e}`);
                // highlightedCode remains escaped HTML
            }
        } else if (typeof Prism !== 'undefined') {
             // Attempt autoloader or default to plaintext
             try {
                 highlightedCode = Prism.highlight(trimmedCode, Prism.languages.plaintext, 'plaintext');
             } catch(e) { /* Already handled by fallback */ }
        }

        // Unique ID for copy button state
        const copyButtonId = `copy-btn-${Date.now()}-${Math.random().toString(16).slice(2)}`;

        return `
            <div class="code-block-container">
                <div class="code-block-header">
                    <span class="code-language">${safeLang}</span>
                    <button class="copy-button" data-copy-target-id="${copyButtonId}" title="Copy code">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="12" height="12"><path fill-rule="evenodd" d="M4.25 2A1.75 1.75 0 0 0 2.5 3.75v8.5A1.75 1.75 0 0 0 4.25 14h8.5A1.75 1.75 0 0 0 14.5 12.25v-8.5A1.75 1.75 0 0 0 12.75 2h-8.5Zm-1.72 1.75a.25.25 0 0 1 .25-.25h8.5a.25.25 0 0 1 .25.25v8.5a.25.25 0 0 1-.25.25h-8.5a.25.25 0 0 1-.25-.25v-8.5Z" clip-rule="evenodd"/></svg>
                        Copy
                    </button>
                </div>
                <pre><code id="${copyButtonId}" class="language-${safeLang}">${highlightedCode}</code></pre>
            </div>
        `;
    }

    function handleSendMessage() {
        if (!messageInput) {return;}
        const message = messageInput.value.trim();
        if (message || attachedFiles.size > 0) {
             logToExtension(`Sending message. Text length: ${message.length}, Files: ${attachedFiles.size}`);
            postMessage({
                type: 'sendMessage',
                message: message,
                attachedFiles: Array.from(attachedFiles) // Send as array
            });
            // User message display is now handled by the 'addMessageRaw' response from backend
            messageInput.value = '';
            attachedFiles.clear(); // Clear attached files after sending
            clearFilePreviews(); // Clear the preview display
            hideAutocompleteList();
            showLoadingIndicator(); // Show loading indicator
            messageInput.style.height = 'auto'; // Reset height
        }
    }

    function handleClearChat() {
        logToExtension("Clear chat button clicked.");
        postMessage({ type: 'clearChat' });
        if (chatContainer) {chatContainer.innerHTML = '';}
        clearFilePreviews(); // Also clear file previews
        attachedFiles.clear();
        if (popupMenu) {popupMenu.classList.remove('show');}
    }

    function handleInput(event) {
        const input = event.target;
        const cursorPosition = input.selectionStart;
        const textBeforeCursor = input.value.substring(0, cursorPosition);

        // Auto-resize textarea
         input.style.height = 'auto'; // Reset height
         input.style.height = `${input.scrollHeight}px`;


        // Autocomplete logic
        const mentionMatch = textBeforeCursor.match(/@([\w\-./\\]*)$/); // Match word chars, -, ., /, \

        if (mentionMatch) {
            const partial = mentionMatch[1];
            const mentionStart = mentionMatch.index;
            const mentionEnd = cursorPosition;
            currentMentionInfo = { start: mentionStart, end: mentionEnd, partial: partial };
            logToExtension(`Requesting autocomplete for partial: "${partial}"`);
            postMessage({ type: 'getAutocompleteSuggestions', partial });
        } else {
            currentMentionInfo = null;
            hideAutocompleteList();
        }
    }

    function handleKeydown(event) {
        // Send on Enter (unless Shift is pressed)
        if (event.key === 'Enter' && !event.shiftKey && !isAutocompleteActive) {
            event.preventDefault();
            handleSendMessage();
        }
        // Autocomplete navigation/selection
        else if (isAutocompleteActive) {
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                event.preventDefault();
                navigateSuggestions(event.key === 'ArrowDown' ? 1 : -1);
            } else if (event.key === 'Tab' || event.key === 'Enter') {
                event.preventDefault();
                selectSuggestion();
            } else if (event.key === 'Escape') {
                event.preventDefault();
                hideAutocompleteList();
            }
        }
    }

    function displayAutocompleteSuggestions(suggestions) {
        if (!autocompleteList || !messageInput) {return;}
        logToExtension(`Displaying ${suggestions.length} suggestions.`);
        currentSuggestions = suggestions; // Store current suggestions
        autocompleteList.innerHTML = ''; // Clear previous suggestions
        selectedSuggestionIndex = -1; // Reset selection

        if (suggestions.length === 0) {
            hideAutocompleteList();
            return;
        }

        suggestions.forEach((suggestion, index) => {
            const item = document.createElement('div');
            item.classList.add('autocomplete-suggestion');
            item.textContent = suggestion;
            item.dataset.index = index; // Store index for navigation

            item.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent body click listener from hiding list immediately
                selectSuggestion(index);
            });
            // Optional: Mouseover selection? Can conflict with keyboard nav.
            // item.addEventListener('mouseover', () => updateSelectionVisual(index));

            autocompleteList.appendChild(item);
        });

        showAutocompleteList();
    }

    function showAutocompleteList() {
        if (!autocompleteList || !inputArea || !messageInput) {return;}
        positionAutocompleteList(); // Position it correctly first
        autocompleteList.style.display = 'block';
        isAutocompleteActive = true;
    }

    function hideAutocompleteList() {
        if (autocompleteList) {autocompleteList.style.display = 'none';}
        isAutocompleteActive = false;
        currentMentionInfo = null;
        currentSuggestions = [];
        selectedSuggestionIndex = -1;
    }

     function positionAutocompleteList() {
         if (!autocompleteList || !inputArea || !messageInput) {return;}
         const inputRect = messageInput.getBoundingClientRect();
         const areaRect = inputArea.getBoundingClientRect();
         // Position above the input area
         autocompleteList.style.bottom = `${areaRect.height}px`;
         autocompleteList.style.left = `${inputRect.left - areaRect.left}px`; // Relative to input area
         autocompleteList.style.width = `${inputRect.width}px`;
     }


    function navigateSuggestions(direction) {
        const items = autocompleteList.children;
        if (!items || items.length === 0) {return;}

        const newIndex = selectedSuggestionIndex + direction;

        // Cycle through suggestions
        if (newIndex >= items.length) {
            selectedSuggestionIndex = 0;
        } else if (newIndex < 0) {
            selectedSuggestionIndex = items.length - 1;
        } else {
            selectedSuggestionIndex = newIndex;
        }
        updateSelectionVisual(selectedSuggestionIndex);
        // Ensure the selected item is visible
        items[selectedSuggestionIndex]?.scrollIntoView?.({ block: 'nearest' });
    }

    // Updates visual state (highlighting) of suggestions
    function updateSelectionVisual(index) {
        const items = autocompleteList.children;
        for (let i = 0; i < items.length; i++) {
            items[i].classList.toggle('selected', i === index);
        }
        selectedSuggestionIndex = index; // Update state
    }

    function selectSuggestion(index = selectedSuggestionIndex) {
        if (index < 0 || index >= currentSuggestions.length || !currentMentionInfo) {
            hideAutocompleteList(); // Invalid selection or no mention active
            return;
        }

        const selectedFile = currentSuggestions[index];
        logToExtension(`Suggestion selected: "${selectedFile}"`);

        // Replace the @mention part with the selected file
        const beforeMention = messageInput.value.substring(0, currentMentionInfo.start);
        const afterMention = messageInput.value.substring(currentMentionInfo.end);
        // Add the file path *after* the @, and maybe a space for continued typing
        const replacementText = `@${selectedFile} `;
        messageInput.value = beforeMention + replacementText + afterMention;

        // Add file to attached list and request preview
        if (!attachedFiles.has(selectedFile)) {
             attachedFiles.add(selectedFile);
             requestFilePreview(selectedFile); // Request preview for the selected file
         }

        hideAutocompleteList(); // Hide after selection
        messageInput.focus(); // Return focus to input

        // Move cursor to the end of the inserted text
        const newCursorPos = currentMentionInfo.start + replacementText.length;
        messageInput.setSelectionRange(newCursorPos, newCursorPos);

         // Trigger input resize after modification
         messageInput.dispatchEvent(new Event('input'));
    }

    function requestFilePreview(filePath) {
        logToExtension(`Requesting file preview for: ${filePath}`);
        postMessage({ type: 'openFilePreviewRequest', filePath });
    }

    function showFilePreview(filePath, content) {
        if (!filePreviewContainer) {return;}
        logToExtension(`Showing file preview for: ${filePath}`);

        // Make container visible if hidden (and has content)
         if (filePreviewContainer.children.length === 0) {
             filePreviewContainer.style.display = 'grid'; // Use grid layout
         }

        // Check if preview already exists
        const previewId = `preview-${filePath.replace(/[^a-zA-Z0-9]/g, '-')}`; // Create safe ID
        if (document.getElementById(previewId)) {
             logToExtension(`Preview already exists for ${filePath}`);
            return; // Don't add duplicate
        }

        const container = document.createElement('div');
        container.classList.add('file-preview');
        container.id = previewId; // Assign ID to the container
        container.dataset.filePath = filePath; // Store file path for removal

        // Simple Preview Structure
        container.innerHTML = `
          <div class="file-header">
            <h4 title="${escapeHtml(filePath)}">${escapeHtml(path.basename(filePath))}</h4>
            <button class="remove-file" data-file-path="${escapeHtml(filePath)}" title="Remove ${escapeHtml(path.basename(filePath))} from context">×</button>
          </div>
          <div class="preview-content-wrapper">
             <pre><code class="language-${escapeHtml(detectLanguage(filePath))}">${escapeHtml(content)}</code></pre>
          </div>
        `;

        filePreviewContainer.appendChild(container);

        // Highlight the code in the new preview
         const codeElement = container.querySelector('code');
         if (codeElement && typeof Prism !== 'undefined') {
             try {
                 Prism.highlightElement(codeElement);
             } catch(e) {
                 logToExtension(`Prism highlighting failed for preview ${filePath}: ${e}`);
             }
         }

        // Add event listener for the remove button (handled by delegation now)
         scrollToBottom(); // Scroll down to show preview if needed
    }

    function clearFilePreviews() {
        if (filePreviewContainer) {
             filePreviewContainer.innerHTML = '';
             filePreviewContainer.style.display = 'none'; // Hide container when empty
         }
     }

    // Basic language detection based on extension
    function detectLanguage(filePath) {
        if (!filePath) {return 'plaintext';}
        const extension = filePath.split('.').pop()?.toLowerCase() || '';
        const languageMap = {
            'js': 'javascript', 'jsx': 'jsx', 'ts': 'typescript', 'tsx': 'tsx',
            'json': 'json', 'html': 'html', 'css': 'css', 'scss': 'scss', 'less': 'less',
            'py': 'python', 'java': 'java', 'c': 'c', 'cpp': 'cpp', 'cs': 'csharp',
            'go': 'go', 'rb': 'ruby', 'php': 'php', 'swift': 'swift', 'kt': 'kotlin',
            'rs': 'rust', 'scala': 'scala', 'sql': 'sql', 'yaml': 'yaml', 'yml': 'yaml',
            'xml': 'xml', 'sh': 'bash', 'bash': 'bash', 'zsh': 'bash',
            'md': 'markdown', 'txt': 'plaintext', '': 'plaintext'
        };
        return languageMap[extension] || 'plaintext';
    }

    // Simplified path.basename for client-side display
     const path = {
         basename: (p) => {
             if (!p) {return '';}
             // Replace backslashes for consistency if needed
             p = p.replace(/\\/g, '/');
             return p.substring(p.lastIndexOf('/') + 1);
         }
     };

    function showLoadingIndicator() {
        if (loadingIndicator) {loadingIndicator.style.display = 'flex';}
        scrollToBottom();
    }

    function hideLoadingIndicator() {
        if (loadingIndicator) {loadingIndicator.style.display = 'none';}
    }

    function scrollToBottom() {
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

     // --- Event Delegation Handler ---
     function handleDynamicClicks(event) {
         const target = event.target;

         // Copy Button
         if (target.closest('.copy-button')) {
             event.preventDefault();
             const button = target.closest('.copy-button');
             const targetId = button.dataset.copyTargetId;
             const codeElement = document.getElementById(targetId);
             if (codeElement) {
                 navigator.clipboard.writeText(codeElement.textContent || '')
                     .then(() => {
                         button.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="12" height="12"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 0 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z" clip-rule="evenodd"/></svg> Copied!`;
                         setTimeout(() => {
                              button.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="12" height="12"><path fill-rule="evenodd" d="M4.25 2A1.75 1.75 0 0 0 2.5 3.75v8.5A1.75 1.75 0 0 0 4.25 14h8.5A1.75 1.75 0 0 0 14.5 12.25v-8.5A1.75 1.75 0 0 0 12.75 2h-8.5Zm-1.72 1.75a.25.25 0 0 1 .25-.25h8.5a.25.25 0 0 1 .25.25v8.5a.25.25 0 0 1-.25.25h-8.5a.25.25 0 0 1-.25-.25v-8.5Z" clip-rule="evenodd"/></svg> Copy`;
                          }, 2000);
                     })
                     .catch(err => {
                         logToExtension(`Clipboard copy failed: ${err}`);
                         button.textContent = 'Error';
                     });
             }
         }
         // Attached File Link (in user message)
         else if (target.classList.contains('attached-file-link')) {
             event.preventDefault();
             const file = target.dataset.file;
             if (file) {
                 logToExtension(`Opening file from link: ${file}`);
                 postMessage({ type: 'openFile', file: file });
             }
         }
          // Remove File Preview Button
          else if (target.classList.contains('remove-file')) {
             event.preventDefault();
             const filePath = target.dataset.filePath;
             const previewElement = target.closest('.file-preview');
             if (filePath && previewElement) {
                 logToExtension(`Removing file preview and attachment: ${filePath}`);
                 previewElement.remove();
                 attachedFiles.delete(filePath);
                 // Hide container if it becomes empty
                 if (filePreviewContainer && filePreviewContainer.children.length === 0) {
                     filePreviewContainer.style.display = 'none';
                 }
                 messageInput.focus(); // Focus input after removing
             }
         }
     }


    // --- Extension Message Handler ---
    function handleExtensionMessage(event) {
        const message = event.data; // The data sent from the extension
        logToExtension(`Message received from extension: ${message.type}`);

        switch (message.type) {
            case 'addMessage': // AI response
                hideLoadingIndicator();
                renderAssistantMessage(message.message);
                break;
            case 'addMessageRaw': // User message echo from backend
                // Note: User messages are added immediately on send now for responsiveness.
                // This handler could be used if we wanted the backend to confirm/format user messages.
                 // For now, we ignore this if it's the user's message echo.
                // if (!message.isUser) { // Only process if it's NOT the user echo
                    hideLoadingIndicator();
                    addMessage(message.html, message.isUser);
                // }
                break;
            case 'autocompleteSuggestions':
                displayAutocompleteSuggestions(message.suggestions || []);
                break;
            case 'restoreHistory':
                 logToExtension(`Restoring history with ${message.messages?.length} messages.`);
                 if (chatContainer) {chatContainer.innerHTML = '';} // Clear existing before restore
                 clearFilePreviews(); // Clear previews on restore too
                 attachedFiles.clear();
                 if (message.messages && Array.isArray(message.messages)) {
                     message.messages.forEach(msg => {
                         if (msg.isUser) {
                             // Need to re-format user message potentially with file links
                             const html = formatUserMessageForDisplay(msg);
                             addMessage(html, true);
                         } else {
                             renderAssistantMessage(msg.message);
                         }
                     });
                 }
                 break;
            case 'showFilePreview':
                showFilePreview(message.filePath, message.content);
                break;
             case 'filePreviewError':
                 // Display error near the file previews? Or as a chat message?
                 logToExtension(`File preview error for ${message.filePath}: ${message.message}`);
                 // Maybe add a small error indicator to the preview container
                 const errorDiv = document.createElement('div');
                 errorDiv.className = 'file-preview-error';
                 errorDiv.textContent = `⚠️ ${path.basename(message.filePath)}: ${message.message}`;
                 if(filePreviewContainer) {filePreviewContainer.prepend(errorDiv);} // Add error at the top
                 break;
             case 'showLoading':
                 showLoadingIndicator();
                 break;
              case 'hideLoading':
                 hideLoadingIndicator();
                 break;
            case 'clearChat': // Message from backend confirming clear
                 if (chatContainer) {chatContainer.innerHTML = '';}
                 clearFilePreviews();
                 attachedFiles.clear();
                 logToExtension("Chat cleared by extension message.");
                 break;
        }
    }

    // Let the extension know the webview is ready
    logToExtension("Webview script loaded and ready.");
    postMessage({ type: 'webviewReady' });

    // Initial setup
    if (messageInput) {
        messageInput.focus();
         messageInput.style.height = 'auto'; // Adjust initial height
         messageInput.style.height = `${messageInput.scrollHeight}px`;
    }
    if (filePreviewContainer) {
         filePreviewContainer.style.display = 'none'; // Ensure hidden initially
     }

})();