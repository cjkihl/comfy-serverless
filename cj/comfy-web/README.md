# @cj/web

Browser test suite and development UI for ComfyUI client testing. This package provides both browser-based testing with Playwright and a React-based UI for interactive testing and debugging.

## Features

- 🧪 **Browser Tests**: Playwright-based end-to-end tests for ComfyUI client
- 🖥️ **Interactive UI**: React-based web interface for testing and debugging
- 🔄 **Dual Test Runners**: Both Bun and Browser test environments
- 📊 **Real Adapter Testing**: Tests using actual WebSocket connections to the proxy
- 🎯 **Session Management Tests**: Validates connection limits and session cleanup
- 🖼️ **Image Testing**: Tests image generation workflows with base64 and binary data

## Prerequisites

Before running tests, ensure you have:

1. **Proxy Server**: Running on `PROXY_URL` (default: `ws://localhost:8190/ws`)
2. **ComfyUI Server**: Running on a remote server (configured via proxy's `PROXY_COMFY_URL`)
3. **Environment Variables**: Set in `.env` file or environment

## Environment Variables

### Required

- `PROXY_COMFY_JWT_SECRET`: **REQUIRED** - HMAC secret used to sign and verify JWT tokens (must match proxy's `PROXY_COMFY_JWT_SECRET`)
- `PROXY_URL`: Proxy WebSocket URL (default: `ws://localhost:8190/ws`)

### Configuration

The package uses `with-env` to automatically load environment variables from `.env` files. Create a `.env` file in the project root:

```bash
# Required
PROXY_COMFY_JWT_SECRET=your-jwt-secret-here

# Optional (defaults shown)
PROXY_URL=ws://localhost:8190/ws
```

## Installation

```bash
# Install dependencies
bun install
```

## Usage

### Development Server

Start the interactive development UI:

```bash
bun run dev
```

This starts a Vite dev server (typically at `http://localhost:5173`) where you can:
- Test ComfyUI client connections
- Submit prompts interactively
- View connection status and logs
- Test image generation workflows

### Running Tests

#### Bun Tests (Server-side)

Tests that run in Bun environment using real WebSocket connections:

```bash
# Run all Bun tests
bun run test:bun

# Run specific test file
bun tests/bun-tests.ts
bun test tests/session-management.spec.ts
```

**Note**: These tests use `with-env` to load environment variables automatically.

#### Browser Tests (Playwright)

End-to-end tests that run in a real browser:

```bash
# Run Playwright tests
bun run test:web

# Run with UI mode (interactive)
bunx playwright test --ui
```

#### All Tests

Run both Bun and browser tests:

```bash
bun run test:all
```

### Test Scripts

The package includes utility scripts for testing:

#### Debug Script

Connect and log all messages from ComfyUI:

```bash
bun run scripts/debug.ts
```

#### Run Test Prompt

Submit a test prompt and save the output image:

```bash
bun run scripts/run-test-prompt.ts
```

Output images are saved to `test-results/` directory.

## Test Structure

### Test Files

- `tests/bun-tests.ts`: Full integration test for Bun environment
- `tests/session-management.spec.ts`: Session management and connection limit tests
- `tests/real-adapter-shared.ts`: Shared test utilities for real adapter testing
- `tests/utils-test.ts`: Test utility functions

### Test Components

- **Connection Tests**: Verify WebSocket connectivity
- **Prompt Submission**: Test prompt submission and execution
- **Event Collection**: Validate event handling and collection
- **Session Management**: Test connection limits and cleanup
- **Concurrent Users**: Test multiple users simultaneously
- **Image Generation**: Test image output (base64 and binary)

## Test Environment Setup

### 1. Start the Proxy

Ensure the proxy server is running:

```bash
cd ../proxy
bun run dev
```

The proxy should be listening on port 8190 (or your configured `PROXY_PORT`).

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
PROXY_COMFY_JWT_SECRET=your-secret-key
PROXY_URL=ws://localhost:8190/ws
```

### 3. Run Tests

```bash
# Run Bun tests
bun run test:bun

# Run browser tests
bun run test:web
```

## Test Output

### Bun Tests

Bun tests output to console with:
- Connection status
- Prompt submission results
- Event collection summaries
- Performance metrics
- Generated images saved to `test-results/`

### Browser Tests

Playwright tests generate:
- Test reports in `test-results/`
- Screenshots on failure
- Video recordings (if configured)
- Test artifacts

## Troubleshooting

### "PROXY_COMFY_JWT_SECRET is not set"

**Solution**: Ensure `PROXY_COMFY_JWT_SECRET` is set in your `.env` file or environment. The package uses `with-env` to load variables automatically.

### "Connection failed"

**Possible causes**:
1. Proxy server is not running
2. `PROXY_URL` is incorrect
3. Proxy's `PROXY_COMFY_JWT_SECRET` doesn't match your `PROXY_COMFY_JWT_SECRET`
4. ComfyUI server is not accessible to the proxy

**Solution**: 
- Verify proxy is running: `curl http://localhost:8190/health`
- Check proxy logs for connection errors
- Ensure `PROXY_COMFY_URL` is correctly configured in proxy

### "Session not ready"

**Possible causes**:
1. ComfyUI server is not running
2. Network connectivity issues
3. Timeout too short

**Solution**:
- Check ComfyUI server status
- Verify network connectivity
- Increase timeout in test configuration

### Tests Timeout

**Possible causes**:
1. ComfyUI server is slow or overloaded
2. Network latency
3. Test timeout too short

**Solution**:
- Increase timeout values in test configuration
- Check ComfyUI server performance
- Verify network conditions

## Development

### Project Structure

```
cj/web/
├── src/                    # Source code
│   ├── components/         # React components
│   ├── hooks/              # React hooks
│   ├── routes/             # React Router routes
│   └── shared/             # Shared utilities
├── tests/                  # Test files
│   ├── bun-tests.ts        # Bun integration tests
│   ├── session-management.spec.ts  # Session tests
│   └── real-adapter-shared.ts      # Shared test utilities
├── scripts/                # Utility scripts
│   ├── debug.ts            # Debug script
│   └── run-test-prompt.ts  # Test prompt runner
└── test-results/           # Test output directory
```

### Adding New Tests

1. Create test file in `tests/` directory
2. Import shared utilities from `tests/real-adapter-shared.ts`
3. Use `generateTestJWT()` for authentication
4. Use `createComfyClient()` helper for client creation
5. Follow existing test patterns

### Type Checking

```bash
bun run type-check
```

### Building

```bash
bun run build
```

## Environment Variable Loading

This package uses `@cjkihl/with-env` to automatically load environment variables. The `with-env` package:

1. Searches for `.env` files starting from the current directory
2. Walks up the directory tree to find `.env` files
3. Loads variables into `process.env`
4. Works in both Bun and Node.js environments

**Important**: Always use `loadEnv()` from `@cjkihl/with-env` at the top of test files and scripts:

```typescript
import { loadEnv } from "@cjkihl/with-env";

await loadEnv();

// Now process.env has all variables loaded
const jwt = generateTestJWT(userId);
```

## Related Packages

- `@cj/comfy-client`: WebSocket client for ComfyUI
- `@cj/comfy-auth`: JWT authentication utilities
- `@cj/comfy-proxy`: WebSocket proxy server

## License

MIT

