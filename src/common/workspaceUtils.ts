// trace\src\workspaceUtils.ts


import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { promisify } from 'util';

const readFile = promisify(fs.readFile);
const stat = promisify(fs.stat);

const MAX_FILE_SIZE = 1024 * 1024; // 1 MB
const MAX_TOTAL_SIZE = 10 * 1024 * 1024; // 10 MB
const MAX_FILES = 100;

export async function getFuzzyFileList(query: string): Promise<string[]> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        return [];
    }

    const config = vscode.workspace.getConfiguration('promptly');
    const ignorePatterns: string[] = config.get('codebaseIgnorePatterns') || [];

    const files = await vscode.workspace.findFiles('**/*', `{${ignorePatterns.join(',')}}`);
    
    return files
        .map(file => vscode.workspace.asRelativePath(file))
        .filter(path => fuzzyMatch(query, path))
        .sort((a, b) => fuzzyScore(query, b) - fuzzyScore(query, a))
        .slice(0, 10); // Limit to 10 results
}

function fuzzyMatch(query: string, str: string): boolean {
    let i = 0, j = 0;
    while (i < query.length && j < str.length) {
        if (query[i].toLowerCase() === str[j].toLowerCase()) {i++;}
        j++;
    }
    return i === query.length;
}

function fuzzyScore(query: string, str: string): number {
    let score = 0;
    let lastIndex = -1;
    for (let i = 0; i < query.length; i++) {
        const index = str.toLowerCase().indexOf(query[i].toLowerCase(), lastIndex + 1);
        if (index === -1) {return 0;}
        score += 1 / (index - lastIndex);
        lastIndex = index;
    }
    return score;
}


export async function getWorkspaceFiles(): Promise<string> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        return "No workspace folder open.";
    }

    const config = vscode.workspace.getConfiguration('promptly');
    const ignorePatterns: string[] = config.get('codebaseIgnorePatterns') || [];

    console.log('Using ignore patterns:', ignorePatterns);

    let fileContents: string[] = [];
    let totalSize = 0;
    let fileCount = 0;

    for (const folder of workspaceFolders) {
        const files = await vscode.workspace.findFiles(
            new vscode.RelativePattern(folder, '**/*'),
            `{${ignorePatterns.join(',')}}`
        );

        for (const file of files) {
            if (fileCount >= MAX_FILES || totalSize >= MAX_TOTAL_SIZE) {
                break;
            }

            try {
                const fileStat = await stat(file.fsPath);
                if (fileStat.size > MAX_FILE_SIZE) {
                    console.log(`Skipping large file: ${file.fsPath}`);
                    continue;
                }

                const content = await readFile(file.fsPath, 'utf8');
                const relativeFilePath = path.relative(folder.uri.fsPath, file.fsPath);
                fileContents.push(`File: ${relativeFilePath}\n${content}\n`);
                
                totalSize += content.length;
                fileCount++;
            } catch (error) {
                console.error(`Error reading file ${file.fsPath}:`, error);
            }
        }

        if (fileCount >= MAX_FILES || totalSize >= MAX_TOTAL_SIZE) {
            break;
        }
    }

    console.log(`Processed ${fileCount} files, total size: ${totalSize} bytes`);
    return fileContents.join('\n---\n');
}


export async function getFileContent(filePath: string): Promise<string> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        throw new Error('No workspace folder open.');
    }

    for (const folder of workspaceFolders) {
        const fullPath = path.join(folder.uri.fsPath, filePath);
        try {
            const content = await readFile(fullPath, 'utf8');
            return content;
        } catch (error) {
            console.error(`Error reading file ${fullPath}:`, error);
        }
    }

    throw new Error(`File not found: ${filePath}`);
}