#!/usr/bin/env python3
"""
LLM Tool Handling Benchmark Script

Tests tool/function calling capabilities across different local LLM models
using Ollama or LM Studio servers.
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
import glob


class BenchmarkConfig:
    """Configuration loaded from .env and config files"""

    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Server configuration
        self.server_type = os.getenv('SERVER_TYPE', 'ollama').lower()
        self.ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = int(os.getenv('OLLAMA_PORT', '11434'))
        self.lmstudio_host = os.getenv('LMSTUDIO_HOST', 'localhost')
        self.lmstudio_port = int(os.getenv('LMSTUDIO_PORT', '1234'))

        # API configuration
        self.api_timeout = int(os.getenv('API_TIMEOUT', '300'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))

        # Model management
        self.auto_pull_models = os.getenv('AUTO_PULL_MODELS', 'true').lower() == 'true'
        self.delete_after_test = os.getenv('DELETE_AFTER_TEST', 'true').lower() == 'true'

        # Results configuration
        self.results_dir = Path(os.getenv('RESULTS_DIR', './results'))

        # Load configuration files
        self.models = self._load_models()
        self.prompts = self._load_prompts()
        self.tools = self._load_tools()
        self.two_stage_prompts = self._load_two_stage_prompts()

    def _load_models(self) -> List[str]:
        """Load model names from config/models.txt"""
        models_file = Path('config/models.txt')
        if not models_file.exists():
            raise FileNotFoundError(f"Models file not found: {models_file}")

        with open(models_file, 'r') as f:
            models = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not models:
            raise ValueError("No models found in config/models.txt")

        return models

    def _load_prompts(self) -> List[str]:
        """Load prompts from config/prompts.txt"""
        prompts_file = Path('config/prompts.txt')
        if not prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not prompts:
            raise ValueError("No prompts found in config/prompts.txt")

        return prompts

    def _load_tools(self) -> List[Dict[str, Any]]:
        """Load tool definitions from config/tools.json"""
        tools_file = Path('config/tools.json')
        if not tools_file.exists():
            raise FileNotFoundError(f"Tools file not found: {tools_file}")

        with open(tools_file, 'r') as f:
            tools = json.load(f)

        if not isinstance(tools, list):
            raise ValueError("tools.json must contain an array of tool definitions")

        if not tools:
            raise ValueError("No tools found in config/tools.json")

        return tools

    def _load_two_stage_prompts(self) -> List[str]:
        """Load two-stage test prompts from config/two_stage_prompts.txt"""
        prompts_file = Path('config/two_stage_prompts.txt')
        if not prompts_file.exists():
            # Two-stage prompts are optional, return empty list if not found
            return []

        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        return prompts

    def get_base_url(self) -> str:
        """Get base URL for the configured server"""
        if self.server_type == 'ollama':
            return f"http://{self.ollama_host}:{self.ollama_port}"
        elif self.server_type == 'lmstudio':
            return f"http://{self.lmstudio_host}:{self.lmstudio_port}"
        else:
            raise ValueError(f"Unknown server type: {self.server_type}")


class ToolSimulator:
    """Simulates tool execution for two-stage testing"""

    @staticmethod
    def validate_path(file_path: str, current_dir: str) -> Dict[str, Any]:
        """Validate that a path is safe and within allowed boundaries"""
        try:
            # Get absolute path
            path = Path(file_path)
            if not path.is_absolute():
                # Make relative paths relative to current_dir
                path = Path(current_dir) / path

            # Resolve to absolute path (resolves .. and symlinks)
            resolved_path = path.resolve()

            # Get the working directory boundary
            working_dir = Path(current_dir).resolve()

            # Check if the path is within the working directory
            try:
                resolved_path.relative_to(working_dir)
            except ValueError:
                return {
                    "valid": False,
                    "error": f"Path '{file_path}' is outside the working directory. Access denied.",
                    "resolved_path": str(resolved_path),
                    "working_directory": str(working_dir)
                }

            return {
                "valid": True,
                "resolved_path": str(resolved_path),
                "working_directory": str(working_dir)
            }

        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid path '{file_path}': {str(e)}",
                "working_directory": current_dir
            }

    @staticmethod
    def list_directory(directory_path: str, recursive: bool = False, pattern: Optional[str] = None, current_dir: str = ".") -> Dict[str, Any]:
        """Simulate the list_directory tool"""
        try:
            # Validate path
            validation = ToolSimulator.validate_path(directory_path, current_dir)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "working_directory": validation["working_directory"]
                }

            path = Path(validation["resolved_path"])
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {directory_path}",
                    "working_directory": current_dir
                }

            if not path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {directory_path}",
                    "working_directory": current_dir
                }

            files = []

            if recursive:
                # Recursive listing
                if pattern:
                    pattern_path = str(path / "**" / pattern)
                    matched_files = glob.glob(pattern_path, recursive=True)
                else:
                    pattern_path = str(path / "**" / "*")
                    matched_files = glob.glob(pattern_path, recursive=True)

                for file_path in matched_files:
                    p = Path(file_path)
                    if p.is_file():
                        files.append({
                            "name": p.name,
                            "path": str(p),
                            "type": "file",
                            "size": p.stat().st_size
                        })
                    elif p.is_dir():
                        files.append({
                            "name": p.name,
                            "path": str(p),
                            "type": "directory"
                        })
            else:
                # Non-recursive listing
                for item in path.iterdir():
                    if pattern and not item.match(pattern):
                        continue

                    if item.is_file():
                        files.append({
                            "name": item.name,
                            "path": str(item),
                            "type": "file",
                            "size": item.stat().st_size
                        })
                    elif item.is_dir():
                        files.append({
                            "name": item.name,
                            "path": str(item),
                            "type": "directory"
                        })

            return {
                "success": True,
                "directory": str(path),
                "files": files,
                "count": len(files),
                "working_directory": current_dir
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "working_directory": current_dir
            }

    @staticmethod
    def read_file(file_path: str, start_line: int = 1, num_lines: Optional[int] = None, current_dir: str = ".") -> Dict[str, Any]:
        """Simulate the read_file tool"""
        try:
            # Validate path
            validation = ToolSimulator.validate_path(file_path, current_dir)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "working_directory": validation["working_directory"]
                }

            path = Path(validation["resolved_path"])
            if not path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "working_directory": current_dir
                }

            if not path.is_file():
                return {
                    "success": False,
                    "error": f"Path is not a file: {file_path}",
                    "working_directory": current_dir
                }

            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Apply line filtering
            start_idx = max(0, start_line - 1)
            if num_lines:
                end_idx = min(len(lines), start_idx + num_lines)
            else:
                end_idx = len(lines)

            content = ''.join(lines[start_idx:end_idx])

            return {
                "success": True,
                "file_path": str(path),
                "content": content,
                "total_lines": len(lines),
                "lines_returned": end_idx - start_idx,
                "working_directory": current_dir
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "working_directory": current_dir
            }

    @staticmethod
    def execute_tool(tool_name: str, parameters: Dict[str, Any], current_dir: str = ".") -> Dict[str, Any]:
        """Execute a tool by name with given parameters"""
        if tool_name == "list_directory":
            return ToolSimulator.list_directory(
                parameters.get("directory_path", "."),
                parameters.get("recursive", False),
                parameters.get("pattern"),
                current_dir
            )
        elif tool_name == "read_file":
            return ToolSimulator.read_file(
                parameters.get("file_path"),
                parameters.get("start_line", 1),
                parameters.get("num_lines"),
                current_dir
            )
        elif tool_name == "write_file":
            # Validate path but don't actually write (read-only simulation)
            validation = ToolSimulator.validate_path(parameters.get("file_path", ""), current_dir)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "working_directory": validation["working_directory"]
                }
            return {
                "success": False,
                "error": "write_file is not supported in benchmark mode (read-only)",
                "working_directory": current_dir
            }
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "working_directory": current_dir
            }


class OllamaClient:
    """Client for Ollama API"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.base_url = config.get_base_url()

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry"""
        print(f"  Pulling model: {model}")
        url = f"{self.base_url}/api/pull"

        try:
            response = requests.post(
                url,
                json={"name": model},
                timeout=self.config.api_timeout,
                stream=True
            )

            # Stream the response to show progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        print(f"    {data['status']}", end='\r')

            print()  # New line after progress
            return True

        except Exception as e:
            print(f"  Error pulling model: {e}")
            return False

    def delete_model(self, model: str) -> bool:
        """Delete a model from Ollama"""
        print(f"  Deleting model: {model}")
        url = f"{self.base_url}/api/delete"

        try:
            response = requests.delete(
                url,
                json={"name": model},
                timeout=30
            )
            return response.status_code == 200

        except Exception as e:
            print(f"  Error deleting model: {e}")
            return False

    def chat(self, model: str, prompt: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send a chat request with tools"""
        url = f"{self.base_url}/api/chat"

        # Get current working directory
        current_dir = os.getcwd()

        # Add system message with working directory context
        system_message = f"You are a helpful assistant with access to tools. Current working directory: {current_dir}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "tools": tools,
            "stream": False
        }

        start_time = time.time()

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.api_timeout
            )
            response.raise_for_status()

            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            result = response.json()

            return {
                "success": True,
                "response": result,
                "response_time_ms": elapsed_time,
                "error": None
            }

        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "response": None,
                "response_time_ms": elapsed_time,
                "error": str(e)
            }

    def chat_multi_turn(self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send a multi-turn chat request with tools"""
        url = f"{self.base_url}/api/chat"

        # Ensure system message with working directory is included
        current_dir = os.getcwd()
        if not messages or messages[0].get('role') != 'system':
            system_message = f"You are a helpful assistant with access to tools. Current working directory: {current_dir}"
            messages = [{"role": "system", "content": system_message}] + messages

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False
        }

        start_time = time.time()

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.api_timeout
            )
            response.raise_for_status()

            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            result = response.json()

            return {
                "success": True,
                "response": result,
                "response_time_ms": elapsed_time,
                "error": None
            }

        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "response": None,
                "response_time_ms": elapsed_time,
                "error": str(e)
            }


class LMStudioClient:
    """Client for LM Studio API (OpenAI-compatible)"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.base_url = config.get_base_url()

    def pull_model(self, model: str) -> bool:
        """LM Studio doesn't support pulling, models must be loaded manually"""
        print(f"  Note: LM Studio requires manual model loading for {model}")
        return True

    def delete_model(self, model: str) -> bool:
        """LM Studio doesn't support deletion via API"""
        print(f"  Note: LM Studio models must be unloaded manually")
        return True

    def chat(self, model: str, prompt: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send a chat request with tools (OpenAI-compatible)"""
        url = f"{self.base_url}/v1/chat/completions"

        # Get current working directory
        current_dir = os.getcwd()

        # Add system message with working directory context
        system_message = f"You are a helpful assistant with access to tools. Current working directory: {current_dir}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "tools": tools,
            "temperature": 0.7
        }

        start_time = time.time()

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.api_timeout
            )
            response.raise_for_status()

            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            result = response.json()

            return {
                "success": True,
                "response": result,
                "response_time_ms": elapsed_time,
                "error": None
            }

        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "response": None,
                "response_time_ms": elapsed_time,
                "error": str(e)
            }

    def chat_multi_turn(self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send a multi-turn chat request with tools (OpenAI-compatible)"""
        url = f"{self.base_url}/v1/chat/completions"

        # Ensure system message with working directory is included
        current_dir = os.getcwd()
        if not messages or messages[0].get('role') != 'system':
            system_message = f"You are a helpful assistant with access to tools. Current working directory: {current_dir}"
            messages = [{"role": "system", "content": system_message}] + messages

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.7
        }

        start_time = time.time()

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.api_timeout
            )
            response.raise_for_status()

            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            result = response.json()

            return {
                "success": True,
                "response": result,
                "response_time_ms": elapsed_time,
                "error": None
            }

        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "response": None,
                "response_time_ms": elapsed_time,
                "error": str(e)
            }


class BenchmarkRunner:
    """Main benchmark runner"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        # Initialize appropriate client
        if config.server_type == 'ollama':
            self.client = OllamaClient(config)
        elif config.server_type == 'lmstudio':
            self.client = LMStudioClient(config)
        else:
            raise ValueError(f"Unsupported server type: {config.server_type}")

        self.results = []
        self.start_time = None
        self.end_time = None

    def run(self):
        """Run the full benchmark"""
        print("=" * 80)
        print("LLM Tool Handling Benchmark")
        print("=" * 80)
        print(f"Server: {self.config.server_type}")
        print(f"Models: {len(self.config.models)}")
        print(f"Standard prompts: {len(self.config.prompts)}")
        print(f"Two-stage prompts: {len(self.config.two_stage_prompts)}")
        print(f"Tools: {len(self.config.tools)}")
        total_tests = (len(self.config.models) * len(self.config.prompts)) + \
                      (len(self.config.models) * len(self.config.two_stage_prompts))
        print(f"Total tests: {total_tests}")
        print("=" * 80)
        print()

        self.start_time = datetime.now()

        # Run benchmark for each model
        for model_idx, model in enumerate(self.config.models, 1):
            print(f"\n[{model_idx}/{len(self.config.models)}] Testing model: {model}")
            print("-" * 80)

            # Pull model if configured
            if self.config.auto_pull_models:
                if not self.client.pull_model(model):
                    print(f"  Skipping model {model} due to pull failure")
                    continue

            # Run standard single-stage prompts
            if self.config.prompts:
                print("\n  Single-Stage Tests:")
                for prompt_idx, prompt in enumerate(self.config.prompts, 1):
                    print(f"\n  [{prompt_idx}/{len(self.config.prompts)}] Prompt: {prompt[:60]}...")

                    result = self.client.chat(model, prompt, self.config.tools)

                    # Extract metrics from response
                    metrics = self._extract_metrics(model, prompt, result)
                    self.results.append(metrics)

                    # Print summary
                    if result['success']:
                        print(f"    ✓ Success | {result['response_time_ms']:.0f}ms | "
                              f"Tool calls: {len(metrics['tool_calls'])}")
                    else:
                        print(f"    ✗ Failed | Error: {result['error']}")

            # Run two-stage tests
            if self.config.two_stage_prompts:
                print("\n  Two-Stage Tests:")
                for prompt_idx, prompt in enumerate(self.config.two_stage_prompts, 1):
                    print(f"\n  [{prompt_idx}/{len(self.config.two_stage_prompts)}] Two-Stage Prompt: {prompt[:60]}...")

                    result = self._run_two_stage_test(model, prompt)
                    self.results.append(result)

                    # Print summary
                    if result['success']:
                        stage1_calls = len(result.get('stage1_tool_calls', []))
                        stage2_calls = len(result.get('stage2_tool_calls', []))
                        print(f"    ✓ Success | {result['response_time_ms']:.0f}ms | "
                              f"Stage1: {stage1_calls} calls, Stage2: {stage2_calls} calls")
                    else:
                        print(f"    ✗ Failed | Error: {result['error']}")

            # Delete model if configured
            if self.config.delete_after_test:
                self.client.delete_model(model)

            print()

        self.end_time = datetime.now()

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

    def _extract_metrics(self, model: str, prompt: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from API response"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt": prompt,
            "test_type": "single_stage",
            "success": result['success'],
            "response_time_ms": result['response_time_ms'],
            "error": result['error'],
            "tool_calls": [],
            "tokens": {
                "input": 0,
                "output": 0,
                "total": 0
            }
        }

        if not result['success'] or not result['response']:
            return metrics

        response = result['response']

        # Extract tool calls (format varies by server)
        if self.config.server_type == 'ollama':
            if 'message' in response and 'tool_calls' in response['message']:
                for tool_call in response['message']['tool_calls']:
                    metrics['tool_calls'].append({
                        "name": tool_call.get('function', {}).get('name'),
                        "parameters": tool_call.get('function', {}).get('arguments', {})
                    })

            # Extract token usage
            if 'prompt_eval_count' in response:
                metrics['tokens']['input'] = response['prompt_eval_count']
            if 'eval_count' in response:
                metrics['tokens']['output'] = response['eval_count']
            metrics['tokens']['total'] = metrics['tokens']['input'] + metrics['tokens']['output']

        elif self.config.server_type == 'lmstudio':
            if 'choices' in response and len(response['choices']) > 0:
                message = response['choices'][0].get('message', {})
                if 'tool_calls' in message:
                    for tool_call in message['tool_calls']:
                        metrics['tool_calls'].append({
                            "name": tool_call.get('function', {}).get('name'),
                            "parameters": json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                        })

            # Extract token usage
            if 'usage' in response:
                usage = response['usage']
                metrics['tokens']['input'] = usage.get('prompt_tokens', 0)
                metrics['tokens']['output'] = usage.get('completion_tokens', 0)
                metrics['tokens']['total'] = usage.get('total_tokens', 0)

        return metrics

    def _run_two_stage_test(self, model: str, prompt: str) -> Dict[str, Any]:
        """Run a two-stage test: initial call, execute tools, follow-up call"""

        total_start_time = time.time()

        # Stage 1: Initial request
        print(f"    Stage 1: Sending initial request...")
        stage1_result = self.client.chat(model, prompt, self.config.tools)

        if not stage1_result['success']:
            return {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt": prompt,
                "test_type": "two_stage",
                "success": False,
                "response_time_ms": stage1_result['response_time_ms'],
                "error": f"Stage 1 failed: {stage1_result['error']}",
                "stage1_tool_calls": [],
                "stage2_tool_calls": [],
                "tool_calls": [],
                "tokens": {"input": 0, "output": 0, "total": 0}
            }

        # Extract stage 1 tool calls
        stage1_metrics = self._extract_metrics(model, prompt, stage1_result)
        stage1_tool_calls = stage1_metrics['tool_calls']

        if not stage1_tool_calls:
            # No tool calls in stage 1, treat as completed
            total_time = (time.time() - total_start_time) * 1000
            return {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt": prompt,
                "test_type": "two_stage",
                "success": True,
                "response_time_ms": total_time,
                "error": "No tool calls in stage 1",
                "stage1_tool_calls": [],
                "stage2_tool_calls": [],
                "tool_calls": [],
                "tokens": stage1_metrics['tokens']
            }

        print(f"    Stage 1: {len(stage1_tool_calls)} tool call(s) detected")

        # Execute tools (simulate)
        current_dir = os.getcwd()
        tool_results = []
        for tool_call in stage1_tool_calls:
            tool_name = tool_call.get('name')
            tool_params = tool_call.get('parameters', {})

            print(f"    Executing tool: {tool_name}({list(tool_params.keys())})")
            result = ToolSimulator.execute_tool(tool_name, tool_params, current_dir)
            tool_results.append({
                "tool": tool_name,
                "result": result
            })

        # Stage 2: Follow-up with tool results
        print(f"    Stage 2: Sending tool results back to model...")

        # Build conversation history with tool results
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Add assistant's response with tool calls (format depends on server)
        if self.config.server_type == 'ollama':
            # Ollama format
            assistant_msg = stage1_result['response'].get('message', {})
            messages.append(assistant_msg)

            # Add tool results as user messages
            for tool_result in tool_results:
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result['result'])
                })

        elif self.config.server_type == 'lmstudio':
            # LM Studio/OpenAI format
            if 'choices' in stage1_result['response'] and len(stage1_result['response']['choices']) > 0:
                assistant_msg = stage1_result['response']['choices'][0]['message']
                messages.append(assistant_msg)

                # Add tool results
                for idx, tool_result in enumerate(tool_results):
                    # Get the tool call ID if available
                    tool_call_id = f"call_{idx}"
                    if 'tool_calls' in assistant_msg and idx < len(assistant_msg['tool_calls']):
                        tool_call_id = assistant_msg['tool_calls'][idx].get('id', tool_call_id)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_result['result'])
                    })

        # Make stage 2 request
        stage2_result = self.client.chat_multi_turn(model, messages, self.config.tools)

        # Extract stage 2 metrics
        if stage2_result['success']:
            stage2_metrics = self._extract_metrics(model, prompt, stage2_result)
            stage2_tool_calls = stage2_metrics['tool_calls']
            print(f"    Stage 2: {len(stage2_tool_calls)} tool call(s) detected")
        else:
            stage2_tool_calls = []
            print(f"    Stage 2: Failed - {stage2_result['error']}")

        # Calculate total metrics
        total_time = (time.time() - total_start_time) * 1000
        total_tokens = {
            "input": stage1_metrics['tokens']['input'] + (stage2_metrics['tokens']['input'] if stage2_result['success'] else 0),
            "output": stage1_metrics['tokens']['output'] + (stage2_metrics['tokens']['output'] if stage2_result['success'] else 0),
            "total": 0
        }
        total_tokens['total'] = total_tokens['input'] + total_tokens['output']

        all_tool_calls = stage1_tool_calls + stage2_tool_calls

        return {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt": prompt,
            "test_type": "two_stage",
            "success": stage1_result['success'] and stage2_result['success'],
            "response_time_ms": total_time,
            "stage1_response_time_ms": stage1_result['response_time_ms'],
            "stage2_response_time_ms": stage2_result['response_time_ms'] if stage2_result['success'] else 0,
            "error": stage2_result['error'] if not stage2_result['success'] else None,
            "stage1_tool_calls": stage1_tool_calls,
            "stage2_tool_calls": stage2_tool_calls,
            "tool_calls": all_tool_calls,
            "tokens": total_tokens,
            "tool_execution_results": tool_results
        }

    def _save_results(self):
        """Save results to JSON files"""
        # Create timestamped directory
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        results_dir = self.config.results_dir / f"results_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Calculate summary statistics
        successful_tests = sum(1 for r in self.results if r['success'])
        failed_tests = len(self.results) - successful_tests
        total_tool_calls = sum(len(r['tool_calls']) for r in self.results)
        avg_response_time = sum(r['response_time_ms'] for r in self.results) / len(self.results) if self.results else 0

        # Group results by model
        results_by_model = {}
        for result in self.results:
            model = result['model']
            if model not in results_by_model:
                results_by_model[model] = []
            results_by_model[model].append(result)

        # Calculate per-model statistics
        model_stats = {}
        for model, model_results in results_by_model.items():
            successful = sum(1 for r in model_results if r['success'])
            failed = len(model_results) - successful
            total_calls = sum(len(r['tool_calls']) for r in model_results)
            avg_time = sum(r['response_time_ms'] for r in model_results) / len(model_results)

            # Separate by test type
            single_stage = [r for r in model_results if r.get('test_type') == 'single_stage']
            two_stage = [r for r in model_results if r.get('test_type') == 'two_stage']

            model_stats[model] = {
                "total_tests": len(model_results),
                "successful_tests": successful,
                "failed_tests": failed,
                "success_rate": round(successful / len(model_results) * 100, 2) if model_results else 0,
                "total_tool_calls": total_calls,
                "avg_response_time_ms": round(avg_time, 2),
                "single_stage_tests": len(single_stage),
                "two_stage_tests": len(two_stage)
            }

            # Add two-stage specific stats if available
            if two_stage:
                successful_two_stage = sum(1 for r in two_stage if r['success'])
                stage1_calls = sum(len(r.get('stage1_tool_calls', [])) for r in two_stage)
                stage2_calls = sum(len(r.get('stage2_tool_calls', [])) for r in two_stage)
                avg_stage1_time = sum(r.get('stage1_response_time_ms', 0) for r in two_stage) / len(two_stage)
                avg_stage2_time = sum(r.get('stage2_response_time_ms', 0) for r in two_stage if r.get('stage2_response_time_ms', 0) > 0)
                if avg_stage2_time > 0:
                    avg_stage2_time = avg_stage2_time / len([r for r in two_stage if r.get('stage2_response_time_ms', 0) > 0])

                model_stats[model]['two_stage_stats'] = {
                    "successful": successful_two_stage,
                    "success_rate": round(successful_two_stage / len(two_stage) * 100, 2),
                    "stage1_tool_calls": stage1_calls,
                    "stage2_tool_calls": stage2_calls,
                    "avg_stage1_time_ms": round(avg_stage1_time, 2),
                    "avg_stage2_time_ms": round(avg_stage2_time, 2) if avg_stage2_time > 0 else 0
                }

        summary = {
            "benchmark_metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds(),
                "server_type": self.config.server_type,
                "server_url": self.config.get_base_url(),
                "total_tests": len(self.results),
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "total_tool_calls": total_tool_calls,
                "avg_response_time_ms": round(avg_response_time, 2)
            },
            "models_tested": self.config.models,
            "prompts_used": self.config.prompts,
            "two_stage_prompts_used": self.config.two_stage_prompts,
            "tools_available": [tool['function']['name'] for tool in self.config.tools],
            "per_model_statistics": model_stats
        }

        # Save summary
        summary_file = results_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed results (all models combined)
        detailed_file = results_dir / "detailed_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save per-model detailed results
        model_files = []
        for model, model_results in results_by_model.items():
            # Sanitize model name for filename
            safe_model_name = model.replace(':', '_').replace('/', '_')
            model_file = results_dir / f"results_{safe_model_name}.json"
            with open(model_file, 'w') as f:
                json.dump(model_results, f, indent=2)
            model_files.append(model_file.name)

        print(f"\n📊 Results saved to: {results_dir}")
        print(f"  - {summary_file.name}")
        print(f"  - {detailed_file.name}")
        for model_file in model_files:
            print(f"  - {model_file}")

    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        successful = sum(1 for r in self.results if r['success'])
        failed = len(self.results) - successful
        total_tools = sum(len(r['tool_calls']) for r in self.results)

        # Separate single-stage and two-stage tests
        single_stage = [r for r in self.results if r.get('test_type') == 'single_stage']
        two_stage = [r for r in self.results if r.get('test_type') == 'two_stage']

        print(f"Total tests: {len(self.results)}")
        print(f"  Single-stage: {len(single_stage)}")
        print(f"  Two-stage: {len(two_stage)}")
        print(f"Successful: {successful} ({successful/len(self.results)*100:.1f}%)")
        print(f"Failed: {failed} ({failed/len(self.results)*100:.1f}%)")
        print(f"Total tool calls: {total_tools}")
        print(f"Duration: {(self.end_time - self.start_time).total_seconds():.1f}s")

        if self.results:
            avg_time = sum(r['response_time_ms'] for r in self.results) / len(self.results)
            print(f"Avg response time: {avg_time:.0f}ms")

        # Two-stage specific metrics
        if two_stage:
            print("\nTwo-Stage Test Metrics:")
            successful_two_stage = sum(1 for r in two_stage if r['success'])
            print(f"  Successful: {successful_two_stage}/{len(two_stage)}")

            stage1_calls = sum(len(r.get('stage1_tool_calls', [])) for r in two_stage)
            stage2_calls = sum(len(r.get('stage2_tool_calls', [])) for r in two_stage)
            print(f"  Stage 1 tool calls: {stage1_calls}")
            print(f"  Stage 2 tool calls: {stage2_calls}")

            avg_stage1_time = sum(r.get('stage1_response_time_ms', 0) for r in two_stage) / len(two_stage)
            avg_stage2_time = sum(r.get('stage2_response_time_ms', 0) for r in two_stage if r.get('stage2_response_time_ms', 0) > 0)
            if avg_stage2_time > 0:
                avg_stage2_time = avg_stage2_time / len([r for r in two_stage if r.get('stage2_response_time_ms', 0) > 0])

            print(f"  Avg Stage 1 time: {avg_stage1_time:.0f}ms")
            if avg_stage2_time > 0:
                print(f"  Avg Stage 2 time: {avg_stage2_time:.0f}ms")

        print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Benchmark tool handling capabilities of local LLMs"
    )
    parser.add_argument(
        '--server',
        choices=['ollama', 'lmstudio'],
        help='Override server type from .env'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = BenchmarkConfig()

        # Override server type if specified
        if args.server:
            config.server_type = args.server

        # Run benchmark
        runner = BenchmarkRunner(config)
        runner.run()

    except FileNotFoundError as e:
        print(f"❌ Configuration error: {e}")
        print("\nMake sure you have:")
        print("  1. Created .env file (copy from .env.example)")
        print("  2. Created config files (models.txt, prompts.txt, tools.json)")
        return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
