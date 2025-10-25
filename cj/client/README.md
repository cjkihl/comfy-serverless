# @cj/comfy-client

A reusable, production-ready WebSocket client for ComfyUI with dual adapter support for both Bun and Browser environments.

## Features

- 🔌 **Dual WebSocket Adapters**: Works in both Bun and Browser environments
- 🔄 **Automatic Reconnection**: Exponential backoff with configurable retry logic
- 🛡️ **Type-Safe**: Full TypeScript support with Result types for error handling
- ⏱️ **Timeout Protection**: Configurable timeouts for all operations
- 📝 **Comprehensive Logging**: Debug, info, warn, error levels with configurable output
- 🎯 **ComfyUI Optimized**: Built-in support for ComfyUI workflows and message types

## Environment Variables

The ComfyClient can be configured using environment variables. All variables have sensible defaults:

### Server Configuration
- `PROXY_URL` - Proxy WebSocket URL for E2E tests (default: `ws://localhost:8190/ws`)
- `PROXY_HTTP_URL` - Proxy HTTP URL (default: `http://localhost:8190`)
- `METRICS_URL` - Metrics endpoint URL (default: `http://localhost:8190/metrics`)
- `METRICS_SECRET` - Metrics authentication secret (default: `123456`)

### Client Configuration
- `CLIENT_TIMEOUT_CONNECT` - Connection timeout in ms (default: `10000`)
- `CLIENT_TIMEOUT_MESSAGE` - Message timeout in ms (default: `30000`)
- `CLIENT_TIMEOUT_OPERATION` - Operation timeout in ms (default: `120000`)

### Reconnection Configuration
- `CLIENT_RECONNECT_ENABLED` - Enable automatic reconnection (default: `true`)
- `CLIENT_RECONNECT_MAX_RETRIES` - Maximum reconnection attempts (default: `5`)
- `CLIENT_RECONNECT_INITIAL_DELAY` - Initial delay in ms (default: `1000`)
- `CLIENT_RECONNECT_MAX_DELAY` - Maximum delay in ms (default: `30000`)
- `CLIENT_RECONNECT_BACKOFF_MULTIPLIER` - Backoff multiplier (default: `2`)

### Logging Configuration
- `CLIENT_LOG_LEVEL` - Log level: `debug`, `info`, `warn`, `error`, `silent` (default: `info`)
- `CLIENT_LOG_PREFIX` - Log prefix (default: `[ComfyClient]`)

### Development Flags
- `DEV_MODE` - Enable development mode (default: `false`)
- `VERBOSE_LOGGING` - Enable verbose logging (default: `false`)

## Quick Start

### Bun Environment

```typescript
import { ComfyClient, BunWebSocketAdapter } from '@cj/comfy-client';

const adapter = new BunWebSocketAdapter();
const client = new ComfyClient({
  url: env.PROXY_URL,
  adapter,
  auth: { jwt: 'your-jwt-token' },
  logging: { level: 'info' }
});

// Connect and submit a prompt
const connectResult = await client.connect();
if (!connectResult.success) {
  console.error('Connection failed:', connectResult.error);
  return;
}

const promptResult = await client.submitPrompt({
  "1": {
    "inputs": { "text": "a beautiful landscape" },
    "class_type": "CLIPTextEncode"
  }
});

if (promptResult.success) {
  console.log('Prompt accepted:', promptResult.data);
}
```

### Browser Environment

```typescript
import { ComfyClient, BrowserWebSocketAdapter } from '@cj/comfy-client';

const adapter = new BrowserWebSocketAdapter();
const client = new ComfyClient({
  url: env.PROXY_URL,
  adapter,
  auth: { jwt: 'your-jwt-token' },
  reconnect: { enabled: true, maxRetries: 5 }
});

await client.connect();
```

### Using Environment Variables

The client automatically uses environment variables for configuration:

```typescript
import { ComfyClient, BunWebSocketAdapter } from '@cj/comfy-client';
import { env } from '@cj/comfy-client/env';

const adapter = new BunWebSocketAdapter();
const client = new ComfyClient({
  url: env.PROXY_URL, // Uses PROXY_URL environment variable
  adapter,
  auth: { jwt: 'your-jwt-token' },
  // All other settings use environment defaults
});

await client.connect();
```

## API Reference

### ComfyClient

The main client class for interacting with ComfyUI via WebSocket.

#### Constructor

```typescript
new ComfyClient(config: ComfyClientConfig)
```

#### Methods

- `connect(): Promise<Result<void>>` - Establish WebSocket connection
- `disconnect(): void` - Close connection and stop reconnection
- `submitPrompt(prompt: ComfyPrompt, options?: SubmitOptions): Promise<Result<PromptAccepted>>` - Submit a ComfyUI prompt
- `waitForEvent(eventType: string, timeout?: number): Promise<Result<unknown>>` - Wait for specific event
- `collectAllEvents(options?: CollectOptions): Promise<Result<EventCollection>>` - Collect all events until completion
- `getConnectionState(): ConnectionState` - Get current connection state
- `isConnected(): boolean` - Check if currently connected

### Configuration

```typescript
type ComfyClientConfig = {
  url: string;                    // WebSocket URL
  adapter: WebSocketAdapter;      // Bun or Browser adapter
  auth?: {                       // Authentication
    jwt?: string;
    apiKey?: string;
  };
  reconnect?: ReconnectConfig;    // Reconnection settings
  timeout?: TimeoutConfig;       // Timeout settings
  logging?: LogConfig;           // Logging configuration
  onMessage?: (msg: unknown) => void;           // Message callback
  onError?: (err: Error) => void;                // Error callback
  onConnectionChange?: (state: ConnectionState) => void; // State callback
};
```

### Result Type

All async operations return a `Result<T>` type for explicit error handling:

```typescript
type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };

// Usage
const result = await client.connect();
if (result.success) {
  console.log('Connected successfully');
} else {
  console.error('Connection failed:', result.error);
}
```

## Error Handling

The client provides specific error types for different failure modes:

- `ComfyConnectionError` - WebSocket connection failures
- `ComfyTimeoutError` - Operation timeouts
- `ComfyAuthError` - Authentication failures
- `ComfyPromptError` - Prompt submission errors

## Examples

See the `examples/` directory for complete usage examples:
- `bun-example.ts` - Server-side usage with Bun
- `browser-example.html` - Client-side browser usage

## License

MIT
