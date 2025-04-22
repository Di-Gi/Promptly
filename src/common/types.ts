// src/common/types.ts
import * as vscode from 'vscode';

// --- API Response Types ---
export interface LocalModelResponse {
    generated_texts: string[];
}

export interface GeminiResponse {
    candidates?: Array<{
        content?: {
            parts?: Array<{
                text?: string;
            }>;
        };
    }>;
}

export interface OpenAIResponse {
    choices: Array<{
        message: {
            content: string;
        };
    }>;
}

export interface AnthropicResponse {
    content: Array<{
        type: string;
        text: string;
    }>;
}

// --- Internal Types ---

export interface ResponsePart {
    type: 'code' | 'text';
    content: string;
    language?: string; // Usually defaults to 'python' if not specified in markdown
}

export interface AnimationControl {
    stop: () => Promise<void>;
    animationRange: vscode.Range;
    insertPosition: vscode.Position; // Position where content will be inserted *after* animation stops
}

export interface CodeBlock {
    code: string;
    language: string | null;
}

// Type for model configuration items in package.json
export type ModelConfigItem = {
    enum?: string[];
    pattern?: string;
};

// Type for server stats
export interface ServerStats {
    cpu_percent: number;
    memory_percent: number;
    queue_size: number;
}

// Types related to MessageRenderer (can be moved to editor/responseRenderer.ts if not shared)
export interface ChunkState {
    text: string;
    isRendered: boolean;
    position: vscode.Position;
    lineContent?: string; // Store the actual line content after rendering
}

export interface MessageRenderState {
    id: string;
    chunks: ChunkState[];
    currentChunkIndex: number;
    isComplete: boolean;
    documentUri: string; // Store URI instead of potentially stale editor object
    decorations: {
        // headerDecoration?: vscode.TextEditorDecorationType; // If needed
        contentDecoration: vscode.TextEditorDecorationType;
    };
    startPosition: vscode.Position; // Store the initial start position
    endPosition?: vscode.Position;   // Store the final end position once known
}