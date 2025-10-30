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

    def get_base_url(self) -> str:
        """Get base URL for the configured server"""
        if self.server_type == 'ollama':
            return f"http://{self.ollama_host}:{self.ollama_port}"
        elif self.server_type == 'lmstudio':
            return f"http://{self.lmstudio_host}:{self.lmstudio_port}"
        else:
            raise ValueError(f"Unknown server type: {self.server_type}")


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

        payload = {
            "model": model,
            "messages": [
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

        payload = {
            "model": model,
            "messages": [
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
        print(f"Prompts: {len(self.config.prompts)}")
        print(f"Tools: {len(self.config.tools)}")
        print(f"Total tests: {len(self.config.models) * len(self.config.prompts)}")
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

            # Run all prompts for this model
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
            "tools_available": [tool['function']['name'] for tool in self.config.tools]
        }

        # Save summary
        summary_file = results_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed results
        detailed_file = results_dir / "detailed_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📊 Results saved to: {results_dir}")
        print(f"  - {summary_file.name}")
        print(f"  - {detailed_file.name}")

    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        successful = sum(1 for r in self.results if r['success'])
        failed = len(self.results) - successful
        total_tools = sum(len(r['tool_calls']) for r in self.results)

        print(f"Total tests: {len(self.results)}")
        print(f"Successful: {successful} ({successful/len(self.results)*100:.1f}%)")
        print(f"Failed: {failed} ({failed/len(self.results)*100:.1f}%)")
        print(f"Total tool calls: {total_tools}")
        print(f"Duration: {(self.end_time - self.start_time).total_seconds():.1f}s")

        if self.results:
            avg_time = sum(r['response_time_ms'] for r in self.results) / len(self.results)
            print(f"Avg response time: {avg_time:.0f}ms")

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
