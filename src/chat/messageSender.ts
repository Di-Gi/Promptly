// src/chat/messageSender.ts
import * as vscode from 'vscode';
import { getCurrentModel } from '../config/modelService';
import { getActivePrompt } from '../config/promptService';
import { sendLocalModelMessage, sendOpenAIMessage, sendAnthropicMessage, sendGeminiMessage } from './apiProviders';
import { OPENAI_GPT_REGEX } from '../common/constants';

/**
 * Sends a message to the appropriate LLM based on the current configuration.
 * Handles API key retrieval and routing.
 */
export async function sendMessage(message: string): Promise<string> {
    const model = getCurrentModel();
    const systemPrompt = getActivePrompt();
    const config = vscode.workspace.getConfiguration('promptly');

    console.log(`Sending message using model: ${model}`);

    try {
        if (model.startsWith('local:')) {
            // Local model doesn't need an API key from settings here
            return await sendLocalModelMessage(message, systemPrompt);
        }

        let apiKey: string | undefined;
        let apiKeyConfig: string;
        let providerName: string;

        if (model.startsWith('gemini-')) {
            apiKeyConfig = 'geminiApiKey';
            providerName = 'Gemini';
            apiKey = config.get(apiKeyConfig) as string;
            if (!apiKey) {throw new Error(`${providerName} API key not configured.`);}
            return await sendGeminiMessage(message, model, apiKey, systemPrompt);

        } else if (model.startsWith('claude-')) {
            apiKeyConfig = 'anthropicApiKey';
            providerName = 'Anthropic';
            apiKey = config.get(apiKeyConfig) as string;
            if (!apiKey) {throw new Error(`${providerName} API key not configured.`);}
            return await sendAnthropicMessage(message, model, apiKey, systemPrompt);

        } else if (model.startsWith('gpt-') || OPENAI_GPT_REGEX.test(model)) {
            apiKeyConfig = 'openaiApiKey';
            providerName = 'OpenAI';
            apiKey = config.get(apiKeyConfig) as string;
            if (!apiKey) {throw new Error(`${providerName} API key not configured.`);}
            return await sendOpenAIMessage(message, model, apiKey, systemPrompt);

        } else {
            throw new Error(`Unsupported model type: ${model}`);
        }

    } catch (error: any) {
        console.error(`Error sending message via ${model}:`, error);
        // Provide a user-friendly error message
        vscode.window.showErrorMessage(`Failed to get response from ${model}: ${error.message}`);
        // Re-throw the error so the caller knows the operation failed
        throw error;
    }
}