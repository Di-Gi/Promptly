// src/editor/responseRenderer.ts
import * as vscode from 'vscode';
import { TextEditor, Position, Range, ThemeColor } from 'vscode';
import { AnimationControl, ChunkState, MessageRenderState } from '../common/types';
import { WHITE_COLOR, BLUE_COLOR } from '../common/constants'; // Use constants
import { getInsertPosition } from './editorService';

const RENDER_DELAY_MS = 15; // Milliseconds between rendering chunks

/**
 * Starts a simple "..." animation in the editor while waiting for a response.
 */
export async function startRequestAnimation(editor: TextEditor, requestPosition: Position): Promise<AnimationControl> {
    let dotCount = 0;
    let animationLine = requestPosition.line + 1; // Start animation on the line below request
    let animationStartPosition: Position;
    let animationRange: Range;

    // Ensure the animation line exists, insert newlines if needed
    await editor.edit(editBuilder => {
        const lineText = editor.document.lineAt(requestPosition.line).text;
        const needsTwoNewlines = lineText.trim().length > 0; // If cursor wasn't at start of line
        editBuilder.insert(requestPosition, needsTwoNewlines ? '\n\n' : '\n');

         // Adjust animation line if newlines were added
        if (needsTwoNewlines) {
             animationLine = requestPosition.line + 2;
         }
    }, { undoStopBefore: true, undoStopAfter: false }); // Group edits if possible

    animationStartPosition = new Position(animationLine, 0);
    animationRange = new Range(animationStartPosition, animationStartPosition.translate(0, 3)); // Initial range for "..."

     const dotDecoration = vscode.window.createTextEditorDecorationType({
         // Use a subtle color, maybe editorInfo foreground or similar
         color: new ThemeColor('editorInfo.foreground'), // Or BLUE_COLOR
         // Optional: Add other styling like italics
         // fontStyle: 'italic'
     });

    let interval: NodeJS.Timeout | undefined = undefined;

    const updateAnimation = () => {
         editor.edit(editBuilder => {
             // Delete previous dots
             const currentRange = new Range(animationStartPosition, animationStartPosition.translate(0, dotCount > 0 ? dotCount : 3));
             editBuilder.delete(currentRange);
             // Insert new dots
             dotCount = (dotCount % 3) + 1;
             editBuilder.insert(animationStartPosition, '.'.repeat(dotCount));
         }, { undoStopBefore: false, undoStopAfter: false }).then(() => {
             // Apply decoration to the new dots
             const newRange = new Range(animationStartPosition, animationStartPosition.translate(0, dotCount));
             // Check if editor is still valid before setting decorations
             if (editor && editor.document && !editor.document.isClosed) {
                  editor.setDecorations(dotDecoration, [newRange]);
             } else {
                 console.warn("Editor closed or invalid during animation update.");
                 stop(); // Attempt to clean up if editor is gone
             }
         }, (err) => {
             console.error("Error during animation edit:", err);
             stop(); // Stop animation on error
         });
     };

    const stop = async () => {
        if (interval) {
            clearInterval(interval);
            interval = undefined;
        }
        dotDecoration.dispose();
        try {
            // Check editor validity before editing
             if (editor && editor.document && !editor.document.isClosed) {
                await editor.edit(editBuilder => {
                    // Use the stored animationRange which covers the max "..." length
                     editBuilder.delete(animationRange);
                }, { undoStopBefore: false, undoStopAfter: true });
             }
        } catch (error) {
            console.error("Error removing animation text:", error);
        }
    };

    // Start the interval
     interval = setInterval(updateAnimation, 500);
     updateAnimation(); // Initial render

    return {
        stop,
        animationRange: animationRange, // The range covering the animation dots
        insertPosition: animationStartPosition // The position where the response should start
    };
}

/**
 * Formats the raw response string for display (e.g., trimming).
 */
export function formatResponse(response: string | undefined): string {
    if (!response) {
        console.error('Received undefined response in formatResponse');
        return 'Error: No response received from the model.';
    }
    // Trim leading/trailing whitespace and newlines
    return response.trim();
}

/**
 * Renders the response message line by line with a slight delay for a streaming effect.
 * Manages decorations and handles potential editor changes during rendering.
 * Uses a Singleton pattern to manage multiple rendering states across documents.
 */
export class MessageRenderer {
    private static instance: MessageRenderer;
    private renderStates: Map<string, MessageRenderState> = new Map(); // Key: document URI string
    private activeRendering: Set<string> = new Set(); // Key: document URI string
    private isDisposed = false;
    private changeListener: vscode.Disposable | undefined;
    private editorChangeListener: vscode.Disposable | undefined;

    private constructor() {
        // Handle potential content changes during rendering
        this.changeListener = vscode.workspace.onDidChangeTextDocument(this.handleDocumentChange.bind(this));
        // Handle switching editors
        this.editorChangeListener = vscode.window.onDidChangeActiveTextEditor(this.handleEditorChange.bind(this));
    }

    static getInstance(): MessageRenderer {
        if (!MessageRenderer.instance) {
            MessageRenderer.instance = new MessageRenderer();
        }
        return MessageRenderer.instance;
    }

     private handleDocumentChange(event: vscode.TextDocumentChangeEvent) {
         const changedUri = event.document.uri.toString();
         if (this.activeRendering.has(changedUri)) {
             console.warn(`Document ${changedUri} changed during rendering. Pausing rendering.`);
             // Potentially pause or stop rendering for this document.
             // For simplicity, we'll let it continue but validation might fail later.
             // More complex handling could involve re-validating the position.
         }
     }

      private handleEditorChange(editor: vscode.TextEditor | undefined) {
         // If the user switches away from an editor that was actively rendering,
         // we might pause or handle it. Currently, validation on resume handles this.
         if (!editor) {return;}
         const activeUri = editor.document.uri.toString();
         const state = this.renderStates.get(activeUri);
         if (state && !state.isComplete && !this.activeRendering.has(activeUri)) {
             console.log(`Resuming rendering for ${activeUri} as it became active.`);
             this.resumeRendering(editor, state);
         }
     }

    private isEditorValidForState(editor: vscode.TextEditor | undefined, state: MessageRenderState): boolean {
        return !!(
            editor &&
            editor.document &&
            !editor.document.isClosed &&
            editor.document.uri.toString() === state.documentUri // Ensure it's the correct document
            // Optional: Check if it's the *active* editor if strict rendering is required
            // && editor === vscode.window.activeTextEditor
        );
    }

    async startRendering(
        editor: vscode.TextEditor,
        response: string,
        startPosition: vscode.Position
    ): Promise<void> {
        if (this.isDisposed) {return;}

        const documentUri = editor.document.uri.toString();
        const formattedResponse = formatResponse(response);
        if (!formattedResponse || formattedResponse.startsWith('Error:')) {
             // Insert error message directly without streaming
             await editor.edit(editBuilder => {
                 editBuilder.insert(startPosition, formattedResponse + '\n');
             });
             return;
         }


        // Clean up any previous state for the same document (optional, depends on desired behavior)
        this.cleanupState(documentUri);

        const state: MessageRenderState = {
            id: `msg-${Date.now()}`,
            chunks: this.createChunks(formattedResponse, startPosition),
            currentChunkIndex: 0,
            isComplete: false,
            documentUri: documentUri,
            startPosition: startPosition,
            decorations: {
                contentDecoration: vscode.window.createTextEditorDecorationType({
                    color: WHITE_COLOR, // Standard text color
                    // Add other styles if needed, e.g., background for the response block
                    // isWholeLine: true,
                    // backgroundColor: new ThemeColor('editor.selectionBackground'),
                }),
            },
        };

        if (state.chunks.length === 0) {
            console.warn("No chunks to render for response.");
            return;
        }

        this.renderStates.set(documentUri, state);
        // Ensure the editor is active before starting, or handle appropriately
         const currentEditor = vscode.window.activeTextEditor;
         if (this.isEditorValidForState(currentEditor, state)) {
            await this.resumeRendering(currentEditor!, state);
         } else {
             console.log(`Editor ${documentUri} is not active. Rendering will start when it becomes active.`);
             // State is stored, will be picked up by handleEditorChange
         }
    }

     private createChunks(text: string, startPosition: vscode.Position): ChunkState[] {
         const lines = text.split('\n');
         const chunks: ChunkState[] = [];
         let currentPosition = new Position(startPosition.line, startPosition.character);

         for (let i = 0; i < lines.length; i++) {
             const lineText = lines[i] + (i < lines.length - 1 ? '\n' : ''); // Add newline except for last line
             if (lineText.length > 0) { // Avoid creating chunks for genuinely empty lines if desired
                 chunks.push({
                     text: lineText,
                     isRendered: false,
                     position: currentPosition // Position where this chunk *starts*
                 });
                 // Calculate position for the *next* chunk
                 if (lineText.endsWith('\n')) {
                     currentPosition = new Position(currentPosition.line + 1, 0);
                 } else {
                     currentPosition = new Position(currentPosition.line, currentPosition.character + lineText.length);
                 }
             } else if (i < lines.length - 1) {
                // Handle intentional empty lines by advancing the position
                currentPosition = new Position(currentPosition.line + 1, 0);
             }
         }
         return chunks;
     }

    private async resumeRendering(editor: vscode.TextEditor, state: MessageRenderState): Promise<void> {
        if (this.isDisposed || this.activeRendering.has(state.documentUri) || state.isComplete) {
            return;
        }

        if (!this.isEditorValidForState(editor, state)) {
             console.log(`Editor ${state.documentUri} is not valid/active for resuming rendering.`);
            return; // Editor changed or closed
        }

        this.activeRendering.add(state.documentUri);
        console.log(`Resuming rendering for ${state.documentUri} from chunk ${state.currentChunkIndex}`);

        try {
            // --- Validation Step ---
            // Check if the content before the current chunk index still matches expectations.
            // This is important if the document could have been edited.
            if (state.currentChunkIndex > 0) {
                const lastRenderedChunk = state.chunks[state.currentChunkIndex - 1];
                 try {
                     // Simple validation: check if the line where the last chunk *ended* exists
                     const endLine = lastRenderedChunk.position.line + (lastRenderedChunk.text.match(/\n/g)?.length || 0);
                     if (endLine >= editor.document.lineCount) {
                          throw new Error("Document content changed significantly (line count mismatch).");
                      }
                      // More complex validation could involve checking actual text content.
                 } catch (validationError: any) {
                     console.error(`Validation failed for ${state.documentUri}: ${validationError.message}. Stopping render.`);
                     this.cleanupState(state.documentUri); // Clean up the invalid state
                     return; // Stop rendering this message
                 }
             }
            // --- End Validation ---


            let cumulativeEdit = new vscode.WorkspaceEdit();
             let editApplied = false;
             let lastPosition: Position | undefined = undefined;

             for (let i = state.currentChunkIndex; i < state.chunks.length; i++) {
                 if (this.isDisposed) {break;} // Check if disposed during loop

                 const chunk = state.chunks[i];
                 const currentEditor = vscode.window.activeTextEditor; // Re-check active editor

                 if (!this.isEditorValidForState(currentEditor, state)) {
                     console.log(`Editor ${state.documentUri} became invalid during rendering loop.`);
                     break; // Stop if editor changed mid-render
                 }

                 // Use WorkspaceEdit for potentially better performance with many small inserts
                 cumulativeEdit.insert(editor.document.uri, chunk.position, chunk.text);
                 chunk.isRendered = true; // Mark as rendered optimistically
                 state.currentChunkIndex = i + 1;
                 lastPosition = chunk.position.translate(
                    (chunk.text.match(/\n/g)?.length || 0),
                     chunk.text.endsWith('\n') ? 0 : chunk.text.length
                 );


                 // Apply the edit and pause
                 try {
                      await vscode.workspace.applyEdit(cumulativeEdit);
                      editApplied = true; // Mark that an edit was applied
                      cumulativeEdit = new vscode.WorkspaceEdit(); // Reset for next iteration
                      await new Promise(resolve => setTimeout(resolve, RENDER_DELAY_MS));
                  } catch (applyError) {
                      console.error(`Error applying edit during rendering for ${state.documentUri}:`, applyError);
                      // Attempt to clean up state on error
                      this.cleanupState(state.documentUri);
                      return; // Stop rendering on apply failure
                  }
             }

            // --- Finalization ---
             if (state.currentChunkIndex >= state.chunks.length) {
                 state.isComplete = true;
                 state.endPosition = lastPosition ?? state.startPosition; // Record the final position
                 console.log(`Rendering complete for ${state.documentUri}`);
                 this.updateDecorations(editor, state);
             }
        } catch (error) {
            console.error(`Error during rendering process for ${state.documentUri}:`, error);
            // Clean up state on unexpected error
            this.cleanupState(state.documentUri);
        } finally {
            this.activeRendering.delete(state.documentUri);
            // If rendering finished or stopped, but wasn't complete, keep the state for potential resume.
             if (state.isComplete) {
                 // Optional: Keep completed states for a while? Or clean up immediately?
                 // For now, let's keep them until explicitly cleared or overwritten.
             }
            console.log(`Finished rendering attempt for ${state.documentUri}`);
        }
    }

     private updateDecorations(editor: vscode.TextEditor, state: MessageRenderState) {
         if (this.isDisposed || !state.isComplete || !state.endPosition) {return;}

         if (!this.isEditorValidForState(editor, state)) {
             return; // Editor not valid
         }

         try {
             const contentRange = new Range(state.startPosition, state.endPosition);
             editor.setDecorations(state.decorations.contentDecoration, [contentRange]);
              console.log(`Applied final decoration for ${state.documentUri} range: ${contentRange.start.line}:${contentRange.start.character} to ${contentRange.end.line}:${contentRange.end.character}`);
         } catch (error) {
             console.error(`Failed to set final decorations for ${state.documentUri}:`, error);
         }
     }


    cleanupState(documentUri: string) {
        const state = this.renderStates.get(documentUri);
        if (state) {
            try {
                 // Attempt to remove decorations from the editor if it's still open
                 const editor = vscode.window.visibleTextEditors.find(e => e.document.uri.toString() === documentUri);
                 if (editor) {
                     editor.setDecorations(state.decorations.contentDecoration, []);
                 }
             } catch (e) {
                 console.error("Error clearing decorations during cleanup:", e);
             } finally {
                 state.decorations.contentDecoration.dispose();
                 this.renderStates.delete(documentUri);
                 this.activeRendering.delete(documentUri); // Ensure it's removed from active set
                 console.log(`Cleaned up render state for ${documentUri}`);
             }
        }
    }

    dispose() {
        console.log("Disposing MessageRenderer...");
        this.isDisposed = true;
        this.changeListener?.dispose();
         this.editorChangeListener?.dispose();
        this.renderStates.forEach((state, uri) => {
            this.cleanupState(uri); // Use cleanup method
        });
        this.renderStates.clear();
        this.activeRendering.clear();
         console.log("MessageRenderer disposed.");
    }
}

/**
 * Appends the response to the file using the MessageRenderer.
 */
export async function appendResponseToFile(
    editor: vscode.TextEditor,
    response: string,
    animationControl: AnimationControl | undefined // Make animation optional
): Promise<void> {
    const startPosition = animationControl ? animationControl.insertPosition : getInsertPosition(editor); // Use animation pos or fallback

    // Stop and remove animation first if it exists
    if (animationControl) {
        try {
            await animationControl.stop();
        } catch (error) {
            console.error("Error stopping animation:", error);
            // Continue trying to render response even if animation removal failed
        }
    }

    try {
        const renderer = MessageRenderer.getInstance();
        // No need to await startRendering directly, it handles its async process
         renderer.startRendering(editor, response, startPosition);
    } catch (error) {
        console.error('Error initiating response rendering:', error);
        vscode.window.showErrorMessage('Failed to start rendering response. Please check logs.');
         // As a fallback, insert the raw response if rendering fails to start
         try {
             await editor.edit(editBuilder => {
                 editBuilder.insert(startPosition, formatResponse(response) + '\n');
             });
         } catch (fallbackError) {
             console.error('Error inserting raw response fallback:', fallbackError);
         }
    }
}