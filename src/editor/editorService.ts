// src/editor/editorService.ts
import * as vscode from 'vscode';
import { TextEditor, Position, Selection } from 'vscode';

/**
 * Gets the prompt text from the editor. Uses selection if available, otherwise the whole document.
 * Returns an empty string if no editor is active.
 */
export function getPrompt(editor: TextEditor | undefined): string {
    if (!editor) {
        return "";
    }
    const selection = editor.selection;
    let prompt: string;
    if (!selection.isEmpty) {
        prompt = editor.document.getText(selection);
    } else {
        // Consider limiting the prompt size if the document is very large
        prompt = editor.document.getText();
    }
    return prompt.trim();
}

/**
 * Determines the position to insert the response.
 * Uses the end of the selection if present, otherwise the cursor position.
 */
export function getInsertPosition(editor: TextEditor): Position {
    const selection = editor.selection;
    // If there's a selection, insert *after* it.
    // If no selection, insert at the current cursor position.
    return selection.end;
}

/**
 * Finds a suitable position to insert a standalone code block,
 * preferably an empty line below the current cursor.
 */
export function findInsertPositionForCodeBlock(editor: TextEditor): Position {
    const document = editor.document;
    const currentPosition = editor.selection.active;

    // Start searching from the line *after* the current cursor line
    for (let line = currentPosition.line + 1; line < document.lineCount; line++) {
        if (document.lineAt(line).isEmptyOrWhitespace) {
            // Found an empty line below, insert at the start of it
            return new Position(line, 0);
        }
    }
    // If no empty line found below, insert at the end of the document on a new line
    const endPosition = new Position(document.lineCount, 0);
    // Check if we need to add a newline before inserting at the very end
    if (document.lineCount > 0 && !document.lineAt(document.lineCount - 1).isEmptyOrWhitespace) {
         // If the last line isn't empty, the insertion will implicitly create a new line
         return endPosition;
    } else {
         // If the last line *is* empty, or doc is empty, just insert there
         return endPosition;
    }
   // Original simpler logic: return new Position(document.lineCount, 0);
}