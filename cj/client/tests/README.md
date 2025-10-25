# ComfyClient Tests

This directory contains comprehensive tests for the `@cj/comfy-client` package.

## Test Structure

- `unit.ts` - Unit tests for individual components
- `integration.ts` - Integration tests for full workflows
- `e2e.ts` - End-to-end tests with real ComfyUI proxy server
- `test-prompt.ts` - Known working ComfyUI test prompt
- `run-test-prompt.ts` - Script to run test prompt and save output images
- `utils.ts` - Test utilities and helpers
- `example.ts` - Simple usage example
- `debug.ts` - Debug tool for message inspection
- `bear-kid.png` - Test image asset used by test prompt

## Running Tests

```bash
# Run all unit tests
bun run test:unit

# Run integration tests (requires ComfyUI server)
bun run test:integration

# Run E2E tests (requires ComfyUI proxy server)
bun run test:e2e

# Run all tests
bun run test:all

# Run test prompt and save output images
bun run test:prompt
```

## Test Categories

### Unit Tests
- WebSocket adapter implementations
- ComfyClient core functionality
- Error handling and Result types
- Configuration validation

### Integration Tests
- Full ComfyUI workflow execution
- Connection lifecycle management
- Reconnection behavior
- Event collection and validation

### E2E Tests
- Single user complete workflows
- Concurrent user testing
- Error handling scenarios
- Performance benchmarking
- Connection pooling validation

## Prerequisites for Integration/E2E Tests

### IMPORTANT: ComfyUI is Running on a Remote Server

⚠️ **ComfyUI is NOT running on localhost** - it's running on a remote server!

Before running tests, ensure you have:

1. **Proxy Server**: Running locally on `env.PROXY_URL` (default: `ws://localhost:8190/ws`)
2. **ComfyUI Server**: Running on **remote server** configured via `COMFY_URL` environment variable
3. **Environment Variables**: Set `COMFY_URL` to point to your remote ComfyUI instance

### Setup Steps

1. **Set the ComfyUI URL** (configure in your proxy's environment):
   ```bash
   export COMFY_URL=http://your-remote-comfyui-server:8188
   ```

2. **Start the proxy server**:
   ```bash
   cd cj/proxy
   bun run dev
   ```

3. **Run tests**:
   ```bash
   cd cj/client
   bun run test:prompt
   ```

**Note**: Tests will fail gracefully if the proxy or ComfyUI is not available. The proxy server handles ComfyUI connectivity and provides appropriate error messages.

## Test Prompt

The `test-prompt.ts` file contains a **known working ComfyUI prompt** that has been tested and verified to work successfully. This prompt:

- Uses SDXL Lightning 2-step model for fast generation
- Includes image input via base64 encoding
- Generates a cute teddy bear drawing
- Outputs high-quality WEBP images
- Has been tested across multiple ComfyUI installations

All tests that expect successful execution use this prompt to ensure reliable test results.

## Running the Test Prompt

To run the test prompt and save the output images for visual comparison:

```bash
# Make sure proxy is running and configured with COMFY_URL
cd cj/client
bun run test:prompt
```

This will:
1. Connect to the ComfyUI proxy server (running locally)
2. Proxy connects to **remote ComfyUI server** (configured via `COMFY_URL`)
3. Submit the test prompt (uses `bear-kid.png` as input)
4. Collect all events and binary data
5. Save output images as `bear-kid-generated-1.webp`, `bear-kid-generated-2.webp`, etc.
6. Print a summary of collected events

The output images will be saved in the `tests/` directory, allowing you to visually compare the generated images with the input image.

### Configuration Notes

- **Proxy Server**: Runs locally, connects to remote ComfyUI
- **COMFY_URL**: Must be set to your remote ComfyUI instance (e.g., `http://remote-server:8188`)
- **PROXY_URL**: Default is `ws://localhost:8190/ws` for client to connect to proxy

## Test Utilities

The `utils.ts` file provides:
- `generateTestJWT(userId)` - Creates unsigned JWT tokens for testing
- `measureTime(fn)` - Measures execution time of async functions
- `createTestPrompt()` - Returns the known working test prompt
- `validateEventSequence(events)` - Validates expected event sequence
- `isValidImageData(data)` - Checks if binary data is a valid image
- `checkComfyHealth(url)` - Checks ComfyUI server health
- `waitForComfyReady(url)` - Waits for ComfyUI to be ready
