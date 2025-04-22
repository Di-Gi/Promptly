// src/config/modelService.ts
import * as vscode from 'vscode';
import { workspace, ConfigurationTarget } from 'vscode';
import { DEFAULT_MODEL, SETUP_LOCAL_MODEL_OPTION } from '../common/constants';
import { ModelConfigItem } from '../common/types';
import { getLocalModelPort } from '../server/serverService'; // Assuming port info might be relevant

let cachedAvailableModels: string[] | null = null;

/**
 * Gets the list of locally configured models (e.g., "local:Qwen/Qwen2.5-7B").
 */
export function getLocalModels(): string[] {
    const config = workspace.getConfiguration('promptly');
    const localModelPath = config.get('localModelPath') as string;
    const localPreconfiguredModel = config.get('localPreconfiguredModel') as string;
    // Check if server is actually running? Maybe not here, responsibility of caller.
    if (localModelPath && localPreconfiguredModel) {
        return [`local:${localPreconfiguredModel}`];
    }
    return [];
}

/**
 * Reads the model list from package.json and adds local models.
 * Caches the result for performance.
 */
export function getAvailableModels(): string[] {
    if (cachedAvailableModels) {
        return cachedAvailableModels;
    }

    const extensionId = 'digi.promptly'; // Replace with your actual extension ID
    const extension = vscode.extensions.getExtension(extensionId);
    let models: string[] = [];

    try {
        if (extension) {
            const packageJSON = extension.packageJSON;
            const modelConfig = packageJSON.contributes?.configuration?.properties?.['promptly.model'];

            if (modelConfig?.oneOf && Array.isArray(modelConfig.oneOf)) {
                const modelEnum = modelConfig.oneOf.find((item: ModelConfigItem) => Array.isArray(item.enum))?.enum;
                if (Array.isArray(modelEnum)) {
                    // Filter out the setup option if it exists in the enum
                    models = modelEnum.filter(model => model !== SETUP_LOCAL_MODEL_OPTION);
                }
            }
        }
    } catch (error) {
        console.error('Error reading model list from package.json:', error);
    }

    if (models.length === 0) {
        console.warn('Unable to retrieve model list from package.json. Using fallback list.');
        // Define a more robust fallback list if necessary
        models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-pro-exp-02-05",
            "o1",
            "o1-mini",
            "o3-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
        ];
    }

    // Add available local models
    const localModels = getLocalModels();
    models = [...models, ...localModels];

    console.log('Available models:', models);
    cachedAvailableModels = models; // Cache the result
    return models;
}

/**
 * Invalidates the cached model list. Call this when local models might have changed.
 */
export function invalidateModelCache(): void {
    cachedAvailableModels = null;
}


/**
 * Gets the currently configured model from settings.
 * Falls back to the first available model or a hardcoded default if the configured one isn't valid.
 */
export function getCurrentModel(): string {
    const config = workspace.getConfiguration('promptly');
    const configuredModel = config.get('model') as string;
    const availableModels = getAvailableModels();

    if (configuredModel && availableModels.includes(configuredModel)) {
        return configuredModel;
    }

    console.warn(`Configured model "${configuredModel}" not found or invalid. Falling back.`);
    // Return the first available model, or the hardcoded default if the list is somehow empty
    return availableModels[0] || DEFAULT_MODEL;
}

/**
 * Checks if the currently selected model in settings is still valid and updates it if not.
 */
export async function updateModelListSetting() {
    const config = workspace.getConfiguration('promptly');
    const currentModel = config.get('model') as string;
    const allModels = getAvailableModels(); // Ensures cache is fresh or regenerated

    if (!allModels.includes(currentModel)) {
        const newModel = allModels[0] || DEFAULT_MODEL;
        console.warn(`Current model "${currentModel}" no longer available. Switching to "${newModel}".`);
        await config.update('model', newModel, ConfigurationTarget.Global);
        vscode.window.showInformationMessage(`The previously selected model "${currentModel}" is no longer available. Switched to: ${newModel}`);
    }
}