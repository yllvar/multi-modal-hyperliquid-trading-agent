# AI Trading Agent

An AI-powered trading system that implements a multi-agent architecture for automated cryptocurrency trading with integrated risk management and execution capabilities.

## Features

- **Multi-Agent Architecture**: Modular design with specialized agents for different trading functions
- **Risk Management**: Comprehensive risk assessment and position sizing
- **Event-Driven**: Asynchronous message passing between components
- **Extensible**: Easy to add new strategies and indicators
- **Backtesting**: Support for historical data testing
- **Real-time Monitoring**: Integrated logging and monitoring

## Agent Components

1. **Base Agent**
   - Core message handling and lifecycle management
   - Common utilities for all agents

2. **Analysis Agent**
   - Processes market data and news
   - Generates trading signals using AI/ML models
   - Implements sentiment analysis

3. **Trading Agent**
   - Manages trading strategies
   - Handles position management
   - Implements order execution logic

4. **Risk Agent**
   - Validates trade requests
   - Implements risk management rules
   - Monitors portfolio risk metrics

5. **Execution Agent**
   - Handles order routing
   - Manages order lifecycle
   - Implements smart order routing

6. **Data Agent** (Planned)
   - Data collection and preprocessing
   - Market data normalization
   - Historical data management

## Table of Contents

- [Disclaimer](#disclaimer)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [License](#license)
- [Disclaimer](#disclaimer-1)
- [Contributing](#contributing)
- [Support](#support)

## Disclaimer

There is no guarantee of any returns. This code has not been audited. Please use at your own risk.

## Architecture

See the full [Architecture Documentation](docs/ARCHITECTURE.md) for detailed information about the system's components, data flow, and design principles.

## Prerequisites

- Python 3.12 or higher
- Docker (for containerized deployment)
- API keys for:
  - Hyperliquid (trading)
  - OpenRouter (LLM access)
  - TAAPI (technical indicators)

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- API keys for required services

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Gajesh2007/ai-trading-agent.git
   cd ai-trading-agent
   ```

2. Set up the virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Update .env with your API keys and configuration
   ```

4. Run the test suite:
   ```bash
   python -m pytest tests/ -v
   ```

## ⚙️ Configuration

### Environment Variables

#### Required
- `MESSAGE_BUS_URL`: URL for the message bus (e.g., `redis://localhost:6379`)
- `LOG_LEVEL`: Logging level (e.g., `INFO`, `DEBUG`)

#### Trading Configuration
- `MAX_POSITION_SIZE`: Maximum position size as percentage of portfolio
- `MAX_RISK_PER_TRADE`: Maximum risk per trade as percentage of portfolio
- `DEFAULT_SLIPPAGE`: Default slippage tolerance

#### API Keys
- `TAAPI_API_KEY`: API key for TAAPI (technical analysis)
- `TOGETHER_API_KEY`: API key for Together.ai (LLM services)

### Obtaining API Keys
- **TAAPI_API_KEY**: Sign up at [TAAPI.io](https://taapi.io/) and generate an API key from your dashboard.
- **HYPERLIQUID_PRIVATE_KEY**: Generate an Ethereum-compatible private key for Hyperliquid. Use tools like MetaMask or `eth_account` library. For security, never share this key.
- **OPENROUTER_API_KEY**: Create an account at [OpenRouter.ai](https://openrouter.ai/), then generate an API key in your account settings.
- **LLM_MODEL**: No key needed; specify a model name like "x-ai/grok-4" (see OpenRouter models list).

## 🏃‍♂️ Running the System

### Start the Trading System
```bash
python run_trading_system.py
```

### Available Commands
- Start all agents: `python run_trading_system.py --all`
- Start specific agent: `python run_trading_system.py --agent trading_agent`
- View help: `python run_trading_system.py --help`

### Monitoring
Access the monitoring dashboard at `http://localhost:3000` (if enabled)

## 🧪 Testing

Run the complete test suite:
```bash
pytest tests/ -v
```

Run with coverage report:
```bash
pytest --cov=agents tests/ -v
```

## 📊 Current Test Coverage

- **Overall Coverage**: 54%
- **Risk Agent**: 64%
- **Trading Agent**: 64%
- **Base Agent**: 79%
- **Execution Agent**: 40%
- **Analysis Agent**: 51%

Docker:
```bash
docker build --platform linux/amd64 -t trading-agent .
docker run --rm -p 3000:3000 --env-file .env trading-agent
# Now: curl http://localhost:3000/diary
```

## Tool Calling
The agent can dynamically fetch any TAAPI indicator (e.g., EMA, RSI) via tool calls. See [TAAPI Indicators](https://taapi.io/indicators/) and [EMA Example](https://taapi.io/indicators/exponential-moving-average/) for details.

## Deployment to EigenCloud

EigenCloud (via EigenX CLI) allows deploying this trading agent in a Trusted Execution Environment (TEE) with secure key management.

### Prerequisites
- Allowlisted Ethereum account (Sepolia for testnet). Request onboarding at [EigenCloud Onboarding](https://onboarding.eigencloud.xyz).
- Docker installed.

### Installation
#### macOS/Linux
```bash
curl -fsSL https://eigenx-scripts.s3.us-east-1.amazonaws.com/install-eigenx.sh | bash
```

#### Windows
```bash
curl -fsSL https://eigenx-scripts.s3.us-east-1.amazonaws.com/install-eigenx.ps1 | powershell -
```

### Initial Setup
```bash
docker login
eigenx auth login  # Or eigenx auth generate --store (if you don't have a eth account, keep this account separate from your trading account)
```

### Deploy the Agent
From the project directory:
```bash
cp .env.example .env
# Edit .env: set ASSETS, INTERVAL, API keys
eigenx app deploy
```

### Monitoring
```bash
eigenx app info --watch
eigenx app logs --watch
```

### Updates
Edit code or .env, then:
```bash
eigenx app upgrade <app-name>
```

For full CLI reference, see the [EigenX Documentation](https://github.com/Layr-Labs/eigenx-cli).
