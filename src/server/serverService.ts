// src/server/serverService.ts
import * as vscode from 'vscode';
import { ServerStats } from '../common/types';
import { invalidateModelCache } from '../config/modelService'; // To update available local models


// --- State ---
// Store the port globally within this module once detected/set.
let localModelPort: number | null = null;

// --- Port Management ---

/**
 * Sets the port number for the local model server.
 * Should be called by localModelSetup when the server starts.
 */
export function setLocalModelPort(port: number): void {
    console.log(`Setting local model server port to: ${port}`);
    localModelPort = port;
    invalidateModelCache(); // Invalidate cache as a local model might now be available
    // Update status bar or context?
}

/**
 * Gets the currently known local model server port.
 * Throws an error if the port is not set (server likely not running or setup incomplete).
 */
export function getLocalModelPort(): number {
    if (localModelPort === null) {
        console.error('Attempted to get local model port, but it was not set.');
        throw new Error('Local model server port not available. Please ensure the server is running (e.g., via "Setup Local Model").');
    }
    return localModelPort;
}

/**
 * Resets the local model port (e.g., on server shutdown).
 */
export function resetLocalModelPort(): void {
    console.log('Resetting local model server port.');
    localModelPort = null;
    invalidateModelCache(); // Invalidate cache as local model is no longer available
}


// --- Server Interaction ---
async function getFetch() {
     try {
        const { default: fetch } = await import('node-fetch');
        return fetch;
    } catch (e) {
        if (typeof fetch !== 'undefined') {return fetch;}
        throw new Error("Fetch API is not available.");
    }
}

// Type guard for server stats response
function isValidServerStats(stats: any): stats is ServerStats {
    return (
        typeof stats === 'object' && stats !== null &&
        typeof stats.cpu_percent === 'number' &&
        typeof stats.memory_percent === 'number' &&
        typeof stats.queue_size === 'number'
    );
}


/**
 * Fetches statistics from the running local model server.
 */
export async function getServerStats(): Promise<ServerStats | null> {
     let port: number;
     try {
         port = getLocalModelPort();
     } catch (error) {
         console.log("Cannot get server stats, port not set.");
         return null; // Server likely not running
     }
     const fetch = await getFetch();
     const url = `http://localhost:${port}/server_stats`;

    try {
        const response = await fetch(url, { signal: AbortSignal.timeout(5000) }); // Short timeout for stats
        if (!response.ok) {
            console.error(`Failed to fetch server stats. Status: ${response.status}`);
            return null;
        }
        const stats = await response.json();
        if (isValidServerStats(stats)) {
            return stats;
        } else {
            console.error('Invalid server stats structure received:', stats);
            return null;
        }
    } catch (error: any) {
        if (error.name !== 'AbortError') { // Don't log timeout errors as server errors
             console.error('Error fetching server stats:', error);
         } else {
              console.log("Fetching server stats timed out (server might be busy or starting).");
          }
        return null;
    }
}

/**
 * Sends a shutdown request to the local model server.
 */
export async function shutdownServer(): Promise<boolean> {
     let port: number;
     try {
         port = getLocalModelPort();
     } catch (error) {
         vscode.window.showErrorMessage('Cannot shutdown server: Port not known. Is it running?');
         return false;
     }
     const fetch = await getFetch();
     const url = `http://localhost:${port}/shutdown`;
     let success = false;

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Shutting down local model server",
        cancellable: false
    }, async (progress) => {
        progress.report({ message: "Sending shutdown request..." });
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                 signal: AbortSignal.timeout(10000) // Timeout for the request itself
            });

            if (!response.ok) {
                console.error(`Failed to send shutdown request. Status: ${response.status}`);
                 vscode.window.showErrorMessage(`Failed to send shutdown request (Status: ${response.status}).`);
                 return; // Exit progress
            }

            console.log('Shutdown request sent successfully');
             progress.report({ message: "Waiting for server confirmation..." });

             // Wait for the server process to actually terminate (poll stats or endpoint)
             let attempts = 0;
             const maxAttempts = 30; // Wait up to 30 seconds
             while (attempts < maxAttempts) {
                 await new Promise(resolve => setTimeout(resolve, 1000));
                 const stats = await getServerStats(); // Use existing stats function to check connectivity
                 if (stats === null) { // If stats returns null, assume server is down
                     console.log('Server appears to have shut down.');
                      progress.report({ increment: 100, message: "Server shut down." });
                      vscode.window.showInformationMessage('Local model server has been shut down.');
                      resetLocalModelPort(); // Crucial: Reset the port state
                      success = true;
                      return; // Exit progress
                  }
                  attempts++;
                  progress.report({ message: `Waiting... (${attempts}/${maxAttempts})` });
              }

              console.error('Server did not shut down within the expected time.');
             vscode.window.showWarningMessage('Server shutdown timed out. It might still be running.');

        } catch (error: any) {
            console.error('Error during server shutdown process:', error);
             if (error.name === 'AbortError') {
                vscode.window.showErrorMessage('Shutdown request timed out.');
             } else if (error.message?.includes('ECONNREFUSED')) {
                  console.log('Server already shut down (connection refused).');
                  progress.report({ increment: 100, message: "Server already shut down." });
                  vscode.window.showInformationMessage('Local model server was already shut down.');
                  resetLocalModelPort(); // Ensure port is reset
                  success = true;
                  return;
             } else {
                vscode.window.showErrorMessage(`Error during server shutdown: ${error.message}`);
             }
        }
    });
     return success;
}


// --- Command Handling (triggered by markerHandler) ---

/**
 * Handles commands prefixed with '??' in the editor. Routes to specific handlers.
 */
export async function handleServerCommand(command: string): Promise<void> {
    const [action, ...args] = command.trim().split(/\s+/); // Split by whitespace

    switch (action.toLowerCase()) {
        case 'status':
            await handleStatusCommand(args);
            break;
        case 'shutdown':
            await handleShutdownCommand(args);
            break;
         case 'port': // Example command
             try {
                 const port = getLocalModelPort();
                 vscode.window.showInformationMessage(`Local model server is running on port: ${port}`);
             } catch (error: any) {
                 vscode.window.showInformationMessage(`Local model server port not set. ${error.message}`);
             }
             break;
        default:
            vscode.window.showInformationMessage(`Unknown server command: "${action}". Available: status, shutdown.`);
    }
}

async function handleStatusCommand(args: string[]): Promise<void> {
    if (args.length === 0 || args[0]?.toLowerCase() === 'server') { // Default to server status
        const stats = await getServerStats();
        if (stats) {
            vscode.window.showInformationMessage(
                `Server Status: CPU: ${stats.cpu_percent.toFixed(1)}%, ` +
                `Memory: ${stats.memory_percent.toFixed(1)}%, ` +
                `Queue: ${stats.queue_size}`
            );
        } else {
            vscode.window.showErrorMessage('Failed to get server stats. Is the local model server running?');
        }
    }
    // Add other status targets here if needed
    // else if (args[0]?.toLowerCase() === 'models') { ... }
    else {
        vscode.window.showInformationMessage('Usage: status [server]');
    }
}

async function handleShutdownCommand(args: string[]): Promise<void> {
     if (args.length === 0 || args[0]?.toLowerCase() === 'server') { // Default to server shutdown
         // Add confirmation dialog
         const confirmation = await vscode.window.showWarningMessage(
             'Are you sure you want to shut down the local model server?',
             { modal: true }, // Make it modal so user must respond
             'Yes, Shutdown'
         );

         if (confirmation === 'Yes, Shutdown') {
             await shutdownServer();
         } else {
              vscode.window.showInformationMessage('Server shutdown cancelled.');
          }
     } else {
         vscode.window.showInformationMessage('Usage: shutdown [server]');
     }
 }