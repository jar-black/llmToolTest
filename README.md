# LLM Tool Handling Benchmark

A comprehensive benchmark script for testing tool/function calling capabilities across different local LLM models using Ollama or LM Studio servers.

## Features

- **Multi-Server Support**: Works with both Ollama and LM Studio
- **Automated Model Management**: Automatically pulls, loads, and deletes models
- **Comprehensive Metrics**: Tracks response time, tool calls, token usage, and success rates
- **Matrix Testing**: Tests every model against every prompt
- **Two-Stage Testing**: Tests sequential tool calling where stage 2 depends on stage 1 results
- **Tool Execution Simulation**: Simulates real tool execution for filesystem operations
- **Flexible Configuration**: Environment-based settings with easy configuration files
- **Detailed Results**: Timestamped JSON output with summary statistics

## Project Structure

```
llm-tool-benchmark/
├── .env                     # Environment configuration (create from .env.example)
├── .env.example            # Example environment template
├── config/
│   ├── models.txt          # One model per line
│   ├── prompts.txt         # One prompt per line (single-stage tests)
│   ├── two_stage_prompts.txt  # Two-stage test prompts (optional)
│   └── tools.json          # Tool definitions (OpenAI format)
├── benchmark.py            # Main benchmark script
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── results/
    └── results_YYYYMMDD_HHMMSS/
        ├── summary.json
        └── detailed_results.json
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd llm-tool-benchmark
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Server Configuration
SERVER_TYPE=ollama          # or 'lmstudio'

# Ollama Server Settings
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# LM Studio Server Settings
LMSTUDIO_HOST=localhost
LMSTUDIO_PORT=1234

# API Configuration
API_TIMEOUT=300
MAX_RETRIES=3

# Model Management
AUTO_PULL_MODELS=true
DELETE_AFTER_TEST=true

# Results
RESULTS_DIR=./results
```

### 4. Configure Test Files

**config/models.txt** - One model per line:
```
llama3.1:8b
mistral:7b
phi3:medium
```

**config/prompts.txt** - One prompt per line:
```
What's the current weather in San Francisco?
Calculate a 15% tip on a $50.00 restaurant bill.
Search for the latest news about artificial intelligence.
```

**config/tools.json** - OpenAI-format tool definitions:
```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a specific location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state"
          }
        },
        "required": ["location"]
      }
    }
  }
]
```

## Usage

### Basic Usage

```bash
# Run with settings from .env
python benchmark.py
```

### Override Server Type

```bash
# Use Ollama
python benchmark.py --server ollama

# Use LM Studio
python benchmark.py --server lmstudio
```

## How It Works

### Workflow

For each model in `config/models.txt`:

1. **Pull Model** (if `AUTO_PULL_MODELS=true`)
   - Downloads model from remote registry
   - Skips if already present

2. **Load Model**
   - Loads model into memory

3. **Run Tests**
   - **Single-Stage Tests**: Sends each prompt from `config/prompts.txt`
   - **Two-Stage Tests**: Sends prompts from `config/two_stage_prompts.txt` (if present)
   - Includes all tools from `config/tools.json` in each request
   - Collects comprehensive metrics

4. **Delete Model** (if `DELETE_AFTER_TEST=true`)
   - Removes model to free up resources
   - Prepares for next model

5. **Save Results**
   - Creates timestamped folder in `results/`
   - Saves summary and detailed results as JSON

### Two-Stage Testing

Two-stage tests evaluate how well models handle sequential tool calls where the second stage depends on the first stage's results:

**Stage 1**: Model receives a prompt and makes initial tool calls (e.g., `list_directory`)
- Example: "Please analyze all files in the ./config directory"
- Model calls: `list_directory(directory_path="./config")`

**Tool Execution**: The benchmark simulates tool execution
- Returns realistic results (actual directory contents)
- Supports `list_directory` and `read_file` tools

**Stage 2**: Model receives tool results and makes follow-up calls
- Model gets the directory listing from Stage 1
- Model should call `read_file` for each file discovered
- Tests the model's ability to use context from previous tool calls

This tests real-world scenarios where multiple tool calls must be coordinated to complete a task.

### Metrics Collected

For each test, the benchmark collects:

- **Response Time**: Total API call duration in milliseconds
- **Tool Calls**: List of tools called with their parameters
- **Token Usage**: Input, output, and total tokens
- **Success/Failure**: Whether the request completed successfully
- **Timestamp**: When the test was executed
- **Error Messages**: Details if test failed

**Additional metrics for two-stage tests:**

- **Stage 1 Tool Calls**: Tools called in the initial request
- **Stage 2 Tool Calls**: Tools called after receiving Stage 1 results
- **Stage 1 Response Time**: Time for initial request
- **Stage 2 Response Time**: Time for follow-up request
- **Tool Execution Results**: Simulated results returned to the model

## Output Format

### Results Directory Structure

```
results/
└── results_20250130_143025/
    ├── summary.json            # High-level statistics
    └── detailed_results.json   # Per-test metrics
```

### Summary JSON Example

```json
{
  "benchmark_metadata": {
    "start_time": "2025-01-30T14:30:25",
    "end_time": "2025-01-30T15:45:12",
    "duration_seconds": 4487.3,
    "server_type": "ollama",
    "server_url": "http://localhost:11434",
    "total_tests": 12,
    "successful_tests": 11,
    "failed_tests": 1,
    "total_tool_calls": 15,
    "avg_response_time_ms": 1250.5
  },
  "models_tested": ["llama3.1:8b", "mistral:7b", "phi3:medium"],
  "prompts_used": ["What's the weather...", "Calculate tip..."],
  "tools_available": ["get_weather", "calculate", "search_news"]
}
```

### Detailed Results JSON Example

**Single-Stage Test:**
```json
{
  "timestamp": "2025-01-30T14:32:10",
  "model": "llama3.1:8b",
  "prompt": "What's the current weather in San Francisco?",
  "test_type": "single_stage",
  "success": true,
  "response_time_ms": 1250.5,
  "error": null,
  "tool_calls": [
    {
      "name": "get_weather",
      "parameters": {
        "location": "San Francisco, CA"
      }
    }
  ],
  "tokens": {
    "input": 150,
    "output": 45,
    "total": 195
  }
}
```

**Two-Stage Test:**
```json
{
  "timestamp": "2025-01-30T14:35:22",
  "model": "llama3.1:8b",
  "prompt": "Please analyze all the files in the ./config directory",
  "test_type": "two_stage",
  "success": true,
  "response_time_ms": 3450.8,
  "stage1_response_time_ms": 1200.3,
  "stage2_response_time_ms": 2250.5,
  "error": null,
  "stage1_tool_calls": [
    {
      "name": "list_directory",
      "parameters": {
        "directory_path": "./config"
      }
    }
  ],
  "stage2_tool_calls": [
    {
      "name": "read_file",
      "parameters": {
        "file_path": "./config/models.txt"
      }
    },
    {
      "name": "read_file",
      "parameters": {
        "file_path": "./config/prompts.txt"
      }
    }
  ],
  "tool_calls": [
    {
      "name": "list_directory",
      "parameters": {
        "directory_path": "./config"
      }
    },
    {
      "name": "read_file",
      "parameters": {
        "file_path": "./config/models.txt"
      }
    },
    {
      "name": "read_file",
      "parameters": {
        "file_path": "./config/prompts.txt"
      }
    }
  ],
  "tokens": {
    "input": 425,
    "output": 180,
    "total": 605
  }
}
```

## Server-Specific Notes

### Ollama

- **Auto-Pull**: Fully supported - models are automatically downloaded
- **Model Deletion**: Supported via API
- **Base URL**: `http://{OLLAMA_HOST}:{OLLAMA_PORT}`
- **API Docs**: https://github.com/ollama/ollama/blob/main/docs/api.md

### LM Studio

- **Auto-Pull**: Not supported - models must be loaded manually via UI
- **Model Deletion**: Not supported via API - unload manually
- **Base URL**: `http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1`
- **Compatibility**: Uses OpenAI-compatible API

## Troubleshooting

### Connection Errors

```bash
# Test Ollama connection
curl http://localhost:11434/api/tags

# Test LM Studio connection
curl http://localhost:1234/v1/models
```

### Model Not Found

- **Ollama**: Enable `AUTO_PULL_MODELS=true` in `.env`
- **LM Studio**: Load model manually in LM Studio UI before running benchmark

### Timeout Errors

Increase timeout in `.env`:
```env
API_TIMEOUT=600  # 10 minutes
```

### Permission Errors

Make script executable:
```bash
chmod +x benchmark.py
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM server
- [LM Studio](https://lmstudio.ai/) - LLM desktop application
- OpenAI - Tool calling API format specification
