// src/chat/apiProviders.ts
import { getLocalModelPort } from '../server/serverService';
import {
    LocalModelResponse, GeminiResponse, OpenAIResponse, AnthropicResponse
} from '../common/types';
import { API_TIMEOUT_MS } from '../common/constants';

// Helper to get fetch dynamically
async function getFetch() {
    try {
        // Use dynamic import for node-fetch
        const { default: fetch } = await import('node-fetch');
        return fetch;
    } catch (e) {
        // Fallback for environments where node-fetch might not be needed or available (like web workers)
        if (typeof fetch !== 'undefined') {
            return fetch;
        }
        console.error("Failed to load 'node-fetch' and native fetch is not available.");
        throw new Error("Fetch API is not available in this environment.");
    }
}


export async function sendLocalModelMessage(message: string, systemPrompt: string): Promise<string> {
    let port: number;
    try {
        port = await getLocalModelPort();
    } catch (error) {
         console.error('Error getting local model port:', error);
         throw new Error('Local model server port not available. Is the server running?');
    }

    const fetch = await getFetch();
    const url = `http://localhost:${port}/generate`;
    const body = JSON.stringify({
        prompt: `${systemPrompt}\n\nUser: ${message}\n\nAssistant:`,
        // Consider making max_length configurable
        max_length: 1024 // Increased default
    });

    console.log(`Sending to local model: ${url}`);
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: body,
            signal: AbortSignal.timeout(API_TIMEOUT_MS) // Use AbortSignal for timeout
        });

        if (!response.ok) {
            let errorDetails = `Status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetails += `, Body: ${JSON.stringify(errorData)}`;
            } catch {
                errorDetails += `, Body: ${await response.text()}`;
            }
             throw new Error(`Local model API request failed. ${errorDetails}`);
        }

        const data = await response.json() as LocalModelResponse;
        console.log('Local model response received.'); // Avoid logging potentially large data

        if (!data || !Array.isArray(data.generated_texts) || data.generated_texts.length === 0 || typeof data.generated_texts[0] !== 'string') {
            console.error('Invalid response format from local model:', data);
            throw new Error('Invalid response format from local model');
        }

        return data.generated_texts[0].trim(); // Trim whitespace
    } catch (error: any) {
        console.error('Error sending message to local model:', error);
        if (error.name === 'AbortError') {
             throw new Error('Request to local model timed out.');
        }
        // Rethrow specific errors or a generic one
        throw new Error(`Failed to communicate with the local model: ${error.message}`);
    }
}

export async function sendOpenAIMessage(message: string, model: string, apiKey: string, systemPrompt: string): Promise<string> {
    const fetch = await getFetch();
    const url = 'https://api.openai.com/v1/chat/completions';
    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
    };
    const body = JSON.stringify({
        model: model,
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: message }
        ],
        // Add other parameters like temperature, max_tokens if needed
    });

    console.log(`Sending to OpenAI model: ${model}`);
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: body,
             signal: AbortSignal.timeout(API_TIMEOUT_MS)
        });

        if (!response.ok) {
             let errorDetails = `Status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetails += `, Body: ${JSON.stringify(errorData)}`;
            } catch {
                 errorDetails += `, Body: ${await response.text()}`;
            }
            throw new Error(`OpenAI API request failed. ${errorDetails}`);
        }

        const data = await response.json() as OpenAIResponse;
         console.log('OpenAI response received.');

        if (!data.choices?.[0]?.message?.content || typeof data.choices[0].message.content !== 'string') {
             console.error('Unexpected response format from OpenAI API:', data);
            throw new Error('Unexpected response format from OpenAI API');
        }

        return data.choices[0].message.content.trim();
    } catch (error: any) {
        console.error('Error sending message to OpenAI:', error);
         if (error.name === 'AbortError') {
             throw new Error('Request to OpenAI timed out.');
        }
        throw new Error(`Failed to communicate with OpenAI API: ${error.message}`);
    }
}

export async function sendAnthropicMessage(message: string, model: string, apiKey: string, systemPrompt: string): Promise<string> {
    const fetch = await getFetch();
    const url = 'https://api.anthropic.com/v1/messages';
    const headers = {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01' // Keep API version updated if necessary
    };
    const body = JSON.stringify({
        model: model,
        max_tokens: 4096, // Increase max tokens, make configurable if needed
        system: systemPrompt, // Use the 'system' parameter
        messages: [
            { role: "user", content: message }
        ]
    });

    console.log(`Sending to Anthropic model: ${model}`);
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: body,
             signal: AbortSignal.timeout(API_TIMEOUT_MS)
        });

        if (!response.ok) {
             let errorDetails = `Status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetails += `, Body: ${JSON.stringify(errorData)}`;
            } catch {
                 errorDetails += `, Body: ${await response.text()}`;
            }
            throw new Error(`Anthropic API request failed. ${errorDetails}`);
        }

        const data = await response.json() as AnthropicResponse;
        console.log('Anthropic response received.');

        // Anthropic returns content as an array of blocks, find the first text block
        const textContent = data.content?.find(block => block.type === 'text');

        if (typeof textContent?.text !== 'string') {
            console.error('Unexpected response format from Anthropic API:', data);
            throw new Error('Unexpected response format from Anthropic API (missing text content)');
        }

        return textContent.text.trim();
    } catch (error: any) {
        console.error('Error sending message to Anthropic:', error);
         if (error.name === 'AbortError') {
             throw new Error('Request to Anthropic timed out.');
        }
        throw new Error(`Failed to communicate with Anthropic API: ${error.message}`);
    }
}

export async function sendGeminiMessage(message: string, model: string, apiKey: string, systemPrompt: string): Promise<string> {
    const fetch = await getFetch();
    // Use v1beta for potentially newer features/models if needed, or v1
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${apiKey}`;
    const headers = {
        'Content-Type': 'application/json',
    };
    // Structure for Gemini API including system instruction
    const body = JSON.stringify({
        contents: [
            { role: "user", parts: [{ text: message }] }
        ],
        systemInstruction: { // Add system prompt here
            parts: [{ text: systemPrompt }]
        },
        generationConfig: { // Make these configurable if desired
            temperature: 0.7,
            topP: 0.95,
            // topK: 64, // Often temperature/topP are sufficient
            maxOutputTokens: 8192,
        },
        // safetySettings: [] // Add safety settings if needed
    });

    console.log(`Sending to Gemini model: ${model}`);
    // Add retries if needed, similar to original code, but simplifying for now
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: body,
             signal: AbortSignal.timeout(API_TIMEOUT_MS)
        });

        if (!response.ok) {
            let errorDetails = `Status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetails += `, Body: ${JSON.stringify(errorData)}`;
            } catch {
                errorDetails += `, Body: ${await response.text()}`;
            }
            throw new Error(`Gemini API request failed. ${errorDetails}`);
        }

        const data = await response.json() as GeminiResponse;
         console.log('Gemini response received.');

        // Robustly access the text part
        const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;

        if (typeof text !== 'string') {
            console.error('Unexpected response format from Gemini API:', data);
            throw new Error('Unexpected response format from Gemini API (missing text content)');
        }

        return text.trim();
    } catch (error: any) {
        console.error('Error sending message to Gemini API:', error);
        if (error.name === 'AbortError') {
             throw new Error('Request to Gemini API timed out.');
        }
        throw new Error(`Failed to communicate with Gemini API: ${error.message}`);
    }
}