// src/config/promptService.ts
import * as vscode from 'vscode';

const DEFAULT_PROMPT_KEY = 'default';

/**
 * Gets the content of the currently active custom prompt from settings.
 * Falls back to the 'default' prompt if the active one isn't found.
 */
export function getActivePrompt(): string {
    const config = vscode.workspace.getConfiguration('promptly');
    const customPrompts = config.get('customPrompts') as { [key: string]: string } | undefined;
    const activePromptName = config.get('activePrompt') as string | undefined;

    if (!customPrompts) {
        console.warn("No custom prompts defined in settings.");
        return "You are a helpful AI assistant."; // Provide a hardcoded absolute fallback
    }

    if (activePromptName && customPrompts[activePromptName]) {
        return customPrompts[activePromptName];
    }

    console.warn(`Active prompt "${activePromptName}" not found. Falling back to default.`);

    if (customPrompts[DEFAULT_PROMPT_KEY]) {
        return customPrompts[DEFAULT_PROMPT_KEY];
    }

    console.warn("Default prompt key not found. Using a hardcoded default prompt.");
    return "You are a helpful AI assistant."; // Fallback if default is also missing
}