// src/common/constants.ts
import * as vscode from 'vscode';

export const RESPONSE_START_MARKER = '\u200B⚡RESPONSE_START⚡\u200B'; // Keep if used for response boundaries
export const RESPONSE_END_MARKER = '\u200B●RESPONSE_END●\u200B';   // Keep if used for response boundaries
export const PROMPT_MARKER = '>>';
export const COMMAND_MARKER = '??';

export const GREEN_COLOR = new vscode.ThemeColor('editorInfo.foreground'); // Or define specific colors if needed
export const BLUE_COLOR = new vscode.ThemeColor('terminalCommandDecoration.defaultBackground'); // Example: Use a different theme color
export const WHITE_COLOR = new vscode.ThemeColor('editor.foreground');

export const DEFAULT_MODEL = 'gemini-2.0-flash'; // Define a default fallback model

export const SETUP_LOCAL_MODEL_OPTION = 'Setup Local Model';

// Regex for OpenAI model names including newer ones like o1, gpt-4o
export const OPENAI_GPT_REGEX = /^(o1|o1-mini|o3-mini|gpt-4o)(-.+)?$/;

// Timeout for API requests (e.g., 5 minutes)
export const API_TIMEOUT_MS = 300000;