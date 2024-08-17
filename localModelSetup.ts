// localModelSetup.ts

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { exec, spawn } from 'child_process';

async function getFetch() {
    return (await import('node-fetch')).default;
}

let outputChannel: vscode.OutputChannel;

function getOutputChannel(): vscode.OutputChannel {
    if (!outputChannel) {
        outputChannel = vscode.window.createOutputChannel('AI Chat Model Server');
    }
    return outputChannel;
}

interface CustomModelConfig {
    id: string;
    path: string;
}

function getRecentCustomModels(): CustomModelConfig[] {
    return vscode.workspace.getConfiguration('promptly').get<CustomModelConfig[]>('recentCustomModels', []);
}

async function saveRecentCustomModel(modelId: string, modelPath: string) {
    const recentModels = getRecentCustomModels();
    const updatedModels = [{ id: modelId, path: modelPath }, ...recentModels.filter(m => m.id !== modelId)].slice(0, 5);
    await vscode.workspace.getConfiguration('promptly').update('recentCustomModels', updatedModels, vscode.ConfigurationTarget.Global);
}

export async function getServerStats(): Promise<{ cpu_percent: number, memory_percent: number, queue_size: number } | null> {
    try {
        const fetch = await getFetch();
        const response = await fetch('http://localhost:8000/server_stats');
        if (response.ok) {
            const stats = await response.json();
            
            // Type guard to check if the response has the expected structure
            if (isValidServerStats(stats)) {
                return {
                    cpu_percent: stats.cpu_percent,
                    memory_percent: stats.memory_percent,
                    queue_size: stats.queue_size
                };
            } else {
                console.error('Invalid server stats structure:', stats);
                return null;
            }
        } else {
            console.error(`Failed to fetch server stats. Status: ${response.status}`);
            return null;
        }
    } catch (error) {
        console.error('Error fetching server stats:', error);
        return null;
    }
}

// Type guard function
function isValidServerStats(stats: any): stats is { cpu_percent: number, memory_percent: number, queue_size: number } {
    return (
        typeof stats === 'object' &&
        stats !== null &&
        typeof stats.cpu_percent === 'number' &&
        typeof stats.memory_percent === 'number' &&
        typeof stats.queue_size === 'number'
    );
}

export async function shutdownServer(): Promise<boolean> {
    let shutdownNotification: vscode.StatusBarItem | undefined;
    try {
        const fetch = await getFetch();
        
        // Create a status bar item for the shutdown process
        shutdownNotification = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
        shutdownNotification.text = "$(sync~spin) Shutting down server...";
        shutdownNotification.show();

        const response = await fetch('http://localhost:8000/shutdown', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            console.log('Shutdown request sent successfully');
            
            // Wait for the server to shut down
            let attempts = 0;
            while (attempts < 30) {  // Wait for up to 30 seconds
                await new Promise(resolve => setTimeout(resolve, 1000));
                try {
                    await fetch('http://localhost:8000/');
                    attempts++;
                    // Update the status bar item
                    shutdownNotification.text = `$(sync~spin) Shutting down server... (${attempts}/30)`;
                } catch (error) {
                    if (error instanceof Error && 'code' in error && error.code === 'ECONNREFUSED') {
                        console.log('Server has shut down');
                        vscode.window.showInformationMessage('Server has shut down successfully');
                        return true;
                    }
                }
            }
            console.error('Server did not shut down within the expected time');
            vscode.window.showErrorMessage('Server shutdown timed out');
            return false;
        } else {
            console.error(`Failed to send shutdown request. Status: ${response.status}`);
            vscode.window.showErrorMessage('Failed to send shutdown request');
            return false;
        }
    } catch (error) {
        console.error('Error sending shutdown request:', error instanceof Error ? error.message : String(error));
        vscode.window.showErrorMessage('Error during server shutdown');
        return false;
    } finally {
        // Always dispose of the status bar item
        if (shutdownNotification) {
            shutdownNotification.dispose();
        }
    }
}

export async function startLocalModelSetup() {
    console.log('Starting local model setup');
    const modelChoice = await selectModelOption();
    if (!modelChoice) {
        console.log('Model selection cancelled');
        vscode.window.showErrorMessage('Model selection cancelled.');
        return;
    }

    let modelConfig: CustomModelConfig;
    if (modelChoice === 'custom') {
        const customConfig = await getCustomModelId();
        if (!customConfig) {
            console.log('Custom model input cancelled');
            vscode.window.showErrorMessage('Custom model input cancelled.');
            return;
        }
        modelConfig = customConfig;
        await saveRecentCustomModel(modelConfig.id, modelConfig.path);
    } else if (modelChoice.startsWith('recent:')) {
        const recentModelId = modelChoice.split(':')[1];
        const recentModel = getRecentCustomModels().find(m => m.id === recentModelId);
        if (!recentModel) {
            console.log('Recent model not found');
            vscode.window.showErrorMessage('Selected recent model not found.');
            return;
        }
        modelConfig = recentModel;
    } else {
        modelConfig = { id: modelChoice, path: vscode.workspace.rootPath || '' };
    }

    const steps = [
        { title: 'Setup Python Environment', execute: setupPythonEnvironment },
        { title: 'Configure Model', execute: () => configureModel(modelConfig.id, modelConfig.path) },
        { title: 'Start Model Server', execute: () => startModelServer(modelConfig.id) }
    ];

    for (const step of steps) {
        console.log(`Executing step: ${step.title}`);
        const result = await step.execute();
        if (!result) {
            console.log(`Setup cancelled during ${step.title} step`);
            vscode.window.showErrorMessage(`Setup cancelled during ${step.title} step.`);
            return;
        }
    }

    console.log('Local model setup complete');
    vscode.window.showInformationMessage('Local model setup complete!');
}



async function getCustomModelId(): Promise<CustomModelConfig | undefined> {
    const modelId = await vscode.window.showInputBox({
        prompt: 'Enter the HuggingFace model ID (e.g., "gpt2", "EleutherAI/gpt-neo-1.3B")',
        placeHolder: 'HuggingFace model ID',
        validateInput: (input) => input.trim() !== '' ? null : 'Please enter a valid model ID'
    });

    if (!modelId) return undefined;

    const modelPath = await vscode.window.showInputBox({
        prompt: `Enter the path where ${modelId} should be installed`,
        value: vscode.workspace.rootPath
    });

    return modelPath ? { id: modelId.trim(), path: modelPath.trim() } : undefined;
}

async function selectModelOption(): Promise<string | undefined> {
    console.log('Entering selectModelOption function');
    const recentCustomModels = getRecentCustomModels();
    
    const options = [
        { label: 'Llama 3', description: 'Latest version of Llama', id: 'meta-llama/Meta-Llama-3.1-8B-Instruct' },
        { label: 'Mixtral', description: 'Mixture of Experts model', id: 'mistralai/Mixtral-8x7B-Instruct-v0.1' },
        { label: 'Custom HuggingFace Model', description: 'Specify your own model', id: 'custom' },
        ...recentCustomModels.map(model => ({
            label: `Recent: ${model.id}`,
            description: `Path: ${model.path}`,
            id: `recent:${model.id}`
        }))
    ];

    const selected = await vscode.window.showQuickPick(options, {
        placeHolder: 'Select a model to install or choose custom',
    });

    console.log('Selected option:', selected?.label);
    return selected?.id;
}
async function setupPythonEnvironment(): Promise<boolean> {
    const pythonExtension = vscode.extensions.getExtension('ms-python.python');
    if (!pythonExtension) {
        vscode.window.showErrorMessage('Python extension is not installed. Please install it and try again.');
        return false;
    }

    await pythonExtension.activate();

    const workspacePath = vscode.workspace.rootPath;
    if (!workspacePath) {
        vscode.window.showErrorMessage('No workspace folder is open. Please open a folder and try again.');
        return false;
    }

    const venvPath = path.join(workspacePath, '.venv');
    const terminal = vscode.window.createTerminal('AI Chat Setup');

    return new Promise<boolean>((resolve) => {
        terminal.show();

        // Create virtual environment
        terminal.sendText(`python -m venv "${venvPath}"`);
        terminal.sendText('');  // Send an empty line to execute the command

        // Activate virtual environment and install packages
        if (process.platform === 'win32') {
            terminal.sendText(`"${path.join(venvPath, 'Scripts', 'activate')}"`);
        } else {
            terminal.sendText(`source "${path.join(venvPath, 'bin', 'activate')}"`);
        }
        terminal.sendText('pip install numpy<2 transformers torch fastapi uvicorn psutil');

        // Add a command to close the terminal
        terminal.sendText('echo "Setup complete. This terminal will close in 60 seconds or you can close it manually."');
        terminal.sendText('echo "Type \'exit\' and press Enter to close this terminal manually."');

        // Set up an event listener to check when the terminal is closed
        const disposable = vscode.window.onDidCloseTerminal(closedTerminal => {
            if (closedTerminal === terminal) {
                disposable.dispose();
                resolve(true);
            }
        });

        // Automatically close the terminal after 60 seconds
        setTimeout(() => {
            terminal.dispose();
        }, 60000);

        vscode.window.showInformationMessage('Setting up Python environment. The terminal will close automatically after 60 seconds, or you can type \'exit\' to close it manually when the setup is complete.');
    });
}

async function configureModel(model: string, modelPath: string): Promise<boolean> {
    console.log('Entering configureModel function with model:', model);
    console.log('Configuring model with path:', modelPath);
    await vscode.workspace.getConfiguration('promptly').update('localModelPath', modelPath, vscode.ConfigurationTarget.Global);
    await vscode.workspace.getConfiguration('promptly').update('localPreconfiguredModel', model, vscode.ConfigurationTarget.Global);
    return true;
}


async function startModelServer(modelName: string): Promise<boolean> {
    console.log('Entering startModelServer function');
    const outputChannel = getOutputChannel();
    outputChannel.show();

    try {
        const pythonPath = await getPythonPath();
        console.log('Using Python path:', pythonPath);
        const modelServerPath = path.join(__dirname, 'model_server.py');

        console.log('Starting model server for model:', modelName);

       // Create or update model_server.py
       const modelServerCode = `
import sys
import traceback
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from pydantic import BaseModel, Field
import torch
import psutil
import asyncio
import signal
from typing import List, Optional
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.queue = asyncio.Queue()
        self.is_processing = False

    async def load_model(self, model_name: str):
        logger.info(f"Starting model download and loading for: {model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = self.model.to(self.device)
            self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, device=self.device)
            logger.info("Model downloaded and loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    async def generate(self, prompt: str, max_length: int, temperature: float, top_p: float, num_return_sequences: int):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
            max_new_tokens = max_length - inputs['input_ids'].shape[1]
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise

    async def process_queue(self):
        if self.is_processing:
            return
        
        self.is_processing = True
        while not self.queue.empty():
            task = await self.queue.get()
            await task
            self.queue.task_done()
        self.is_processing = False

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_name = "${modelName}"  # This will be replaced with the huggingface path
    try:
        await model_manager.load_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)
    yield
    # Shutdown
    logger.info("Shutting down the model server")

app = FastAPI(lifespan=lifespan)
shutdown_event = asyncio.Event()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = Field(default=100, ge=1, le=1000)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    num_return_sequences: int = Field(default=1, ge=1, le=5)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Model server is running"}

@app.post("/generate")
async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
    if model_manager.pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    async def generate_text():
        try:
            generated_texts = await model_manager.generate(
                request.prompt,
                request.max_length,
                request.temperature,
                request.top_p,
                request.num_return_sequences
            )
            logger.info(f"Generated {len(generated_texts)} sequences")
            return {"generated_texts": generated_texts}
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    task = asyncio.create_task(generate_text())
    await model_manager.queue.put(task)
    background_tasks.add_task(model_manager.process_queue)
    
    return await task

@app.get("/model_info")
async def model_info():
    if model_manager.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": model_manager.model.config.name_or_path,
        "model_type": model_manager.model.config.model_type,
        "vocab_size": model_manager.model.config.vocab_size,
        "hidden_size": model_manager.model.config.hidden_size,
        "num_layers": model_manager.model.config.num_hidden_layers,
        "num_heads": model_manager.model.config.num_attention_heads,
        "device": model_manager.device
    }

@app.get("/server_stats")
async def server_stats():
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": memory.percent,
        "memory_available": memory.available,
        "queue_size": model_manager.queue.qsize()
    }

@app.post("/shutdown")
async def shutdown():
    logger.info("Shutdown requested")
    shutdown_event.set()
    return JSONResponse({"message": "Server is shutting down"})

if __name__ == "__main__":
    import uvicorn
    
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, lifespan="on")
    server = uvicorn.Server(config)
    
    async def run_server_with_graceful_shutdown():
        server_task = asyncio.create_task(server.serve())
        await shutdown_event.wait()
        logger.info("Shutdown event received, stopping the server")
        server.should_exit = True
        await server.shutdown()
        await server_task
        logger.info("Server shutdown complete")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_server_with_graceful_shutdown())
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        group = asyncio.gather(*pending, return_exceptions=True)
        loop.run_until_complete(group)
        loop.close()
        logger.info("Shutdown process completed")
`;
       
               fs.writeFileSync(modelServerPath, modelServerCode);

        return new Promise<boolean>((resolve) => {
            let serverProcess: ReturnType<typeof spawn>;
            try {
                console.log('Spawning Python process with:', pythonPath, modelServerPath);
                serverProcess = spawn(pythonPath, [modelServerPath]);
            } catch (error) {
                const errorMessage = `Failed to spawn Python process: ${error}`;
                console.error(errorMessage);
                outputChannel.appendLine(errorMessage);
                vscode.window.showErrorMessage(errorMessage);
                resolve(false);
                return;
            }

            if (serverProcess.stdout) {
                serverProcess.stdout.on('data', (data) => {
                    const output = data.toString();
                    console.log('Server stdout:', output);
                    outputChannel.appendLine(output);
                    if (output.includes('Starting server...')) {
                        vscode.window.showInformationMessage('Model server is starting...');
                    }
                    if (output.includes('Model downloaded and loaded successfully')) {
                        vscode.window.showInformationMessage('Model downloaded and loaded successfully.');
                    }
                });
            } else {
                console.warn('Server process stdout is null');
                outputChannel.appendLine('Warning: Unable to capture server stdout');
            }

            if (serverProcess.stderr) {
                serverProcess.stderr.on('data', (data) => {
                    const error = data.toString();
                    console.error('Server stderr:', error);
                    outputChannel.appendLine(`ERROR: ${error}`);
                    if (error.includes('Error:')) {
                        vscode.window.showErrorMessage(`Model server error: ${error}`);
                    }
                });
            } else {
                console.warn('Server process stderr is null');
                outputChannel.appendLine('Warning: Unable to capture server stderr');
            }

            serverProcess.on('error', (error) => {
                const errorMessage = `Server process error: ${error.message}`;
                console.error(errorMessage);
                outputChannel.appendLine(errorMessage);
                vscode.window.showErrorMessage(errorMessage);
                resolve(false);
            });

            serverProcess.on('close', (code) => {
                console.log('Server process closed with code:', code);
                if (code !== 0) {
                    const errorMessage = `Server process exited with code ${code}`;
                    outputChannel.appendLine(errorMessage);
                    vscode.window.showErrorMessage('Model server failed to start. Check the output channel for details.');
                    resolve(false);
                } else {
                    outputChannel.appendLine('Server process closed successfully');
                    resolve(true);
                }
            });

            setTimeout(async () => {
                try {
                    const fetch = await getFetch();
                    const response = await fetch('http://localhost:8000/');
                    if (response.ok) {
                        console.log('Model server is running');
                        vscode.window.showInformationMessage('Model server is running.');
                        resolve(true);
                    } else {
                        throw new Error(`Server responded with status: ${response.status}`);
                    }
                } catch (error: unknown) {
                    if (error instanceof Error) {
                        if (error.message.includes('ECONNREFUSED')) {
                            console.error('Server is not running yet. Waiting longer...');
                        } else {
                            const errorMessage = `Error checking server status: ${error.message}`;
                            console.error(errorMessage);
                            outputChannel.appendLine(errorMessage);
                            vscode.window.showWarningMessage('Model server may not have started properly. Check the output channel for details.');
                            serverProcess.kill();
                            resolve(false);
                        }
                    } else {
                        console.error('An unknown error occurred');
                        outputChannel.appendLine('An unknown error occurred');
                        vscode.window.showWarningMessage('An unknown error occurred while checking the server status.');
                        serverProcess.kill();
                        resolve(false);
                    }
                }
            }, 120000); // 2 minute timeout

            vscode.window.showInformationMessage('Starting model server. This may take a few minutes...');
        });
    } catch (error) {
        const errorMessage = `Error in startModelServer: ${error}`;
        console.error(errorMessage);
        outputChannel.appendLine(errorMessage);
        vscode.window.showErrorMessage(errorMessage);
        return false;
    }
}

async function getPythonPath(): Promise<string> {
    try {
        const pythonExtension = vscode.extensions.getExtension('ms-python.python');
        if (pythonExtension) {
            await pythonExtension.activate();
            const pythonPath = pythonExtension.exports.settings.getExecutionDetails().execCommand;
            if (pythonPath) {
                // If pythonPath is an array, return the first element
                return Array.isArray(pythonPath) ? pythonPath[0] : pythonPath;
            }
        }
    } catch (error) {
        console.error('Error getting Python path from extension:', error);
    }

    // Fallback to system Python
    return new Promise((resolve, reject) => {
        exec('python -c "import sys; print(sys.executable)"', (error, stdout, stderr) => {
            if (error) {
                reject(`Error getting Python path: ${error.message}`);
                return;
            }
            if (stderr) {
                reject(`Error getting Python path: ${stderr}`);
                return;
            }
            resolve(stdout.trim());
        });
    });
}