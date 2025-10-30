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
