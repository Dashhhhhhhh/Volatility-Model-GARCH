# BTC GARCH Volatility Model #
## Quick Start

### Using Poetry

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Copy environment variables:
   ```bash
   cp .env.example .env
   ```

3. Run tests:
   ```bash
   poetry run pytest
   ```

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t btc-garch .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 btc-garch
   ```


MIT License
