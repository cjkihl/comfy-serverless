# @cj/comfy-client

A reusable, production-ready WebSocket client for ComfyUI with universal adapter support for both Bun and Browser environments.

## Features

- 🔌 **Universal WebSocket Adapter**: Works in both Bun and Browser environments with a single adapter
- 🔄 **Automatic Reconnection**: Exponential backoff with configurable retry logic
- 🛡️ **Type-Safe**: Full TypeScript support with Result types for error handling
- ⏱️ **Timeout Protection**: Configurable timeouts for all operations
- 📝 **Comprehensive Logging**: Debug, info, warn, error levels with configurable output
- 🎯 **ComfyUI Optimized**: Built-in support for ComfyUI workflows and message types
- 🔐 **Query Parameter Auth**: Automatically handles JWT authentication via URL query parameters (WebSockets don't support custom headers)
- ⚡ **Auto-Connect**: Optional automatic connection on client creation
- 💓 **Heartbeat**: Automatic ping/pong to keep connections alive
- 👁️ **Tab Focus Reconnection**: Automatically reconnects when browser tab regains focus

## Quick Start

### Bun Environment

```typescript
import { ComfyClient } from '@cj/comfy-client';
import { UniversalWebSocketAdapter } from '@cj/comfy-client/adapters/universal';

const adapter = new UniversalWebSocketAdapter();
const client = new ComfyClient({
  url: 'ws://localhost:8190/ws',
  adapter,
  auth: { jwt: 'your-jwt-token' },
  logging: { level: 'info' },
  autoConnect: true // Automatically connect on creation
});

// If autoConnect is false, manually connect:
// const connectResult = await client.connect();
// if (!connectResult.success) {
//   console.error('Connection failed:', connectResult.error);
//   return;
// }

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
import { ComfyClient } from '@cj/comfy-client';
import { UniversalWebSocketAdapter } from '@cj/comfy-client/adapters/universal';

const adapter = new UniversalWebSocketAdapter();
const client = new ComfyClient({
  url: 'ws://localhost:8190/ws',
  adapter,
  auth: { jwt: 'your-jwt-token' },
  reconnect: { enabled: true, maxRetries: 5 },
  autoConnect: true // Automatically connect on creation
});

// Client will automatically connect and reconnect on tab focus
```

## API Reference

### ComfyClient

The main client class for interacting with ComfyUI via WebSocket.

#### Constructor

```typescript
new ComfyClient(config: ComfyClientConfig)
```

#### Methods

- `connect(): Promise<Result<void>>` - Establish WebSocket connection (automatically called if `autoConnect: true`)
- `disconnect(): void` - Close connection and stop reconnection
- `submitPrompt(prompt: ComfyPrompt, options?: SubmitOptions): Promise<Result<unknown>>` - Submit a ComfyUI prompt
- `waitForEvent(eventType: string, timeout?: number): Promise<Result<unknown>>` - Wait for specific event
- `collectAllEvents(options?: CollectOptions): Promise<Result<EventCollection>>` - Collect all events until completion
- `ping(): Promise<Result<void>>` - Send a ping message to the server
- `getConnectionState(): ConnectionState` - Get current connection state
- `isConnected(): boolean` - Check if currently connected
- `validateEventSequence(events: unknown[]): { valid: boolean; missingEvents: string[]; extraEvents: string[] }` - Validate event sequence (useful for testing)

### Configuration

```typescript
type ComfyClientConfig = {
  url: string;                    // WebSocket URL
  adapter: WebSocketAdapter;      // Universal adapter (works in Bun and Browser)
  auth: {                         // Authentication (required)
    jwt: string;
  };
  autoConnect?: boolean;           // Automatically connect on creation (default: true)
  heartbeat?: Partial<HeartbeatConfig>;  // Heartbeat settings
  reconnect?: Partial<ReconnectConfig>;  // Reconnection settings
  timeout?: Partial<TimeoutConfig>;      // Timeout settings
  logging?: Partial<LogConfig>;         // Logging configuration
  onMessage?: (msg: ComfyMessage) => void;              // Message callback
  onError?: (err: ComfyClientError) => void;            // Error callback
  onConnectionChange?: (state: ConnectionState) => void; // State callback
};

type HeartbeatConfig = {
  enabled: boolean;    // Enable heartbeat (default: true)
  interval: number;    // Heartbeat interval in ms (default: 30000)
};

type ReconnectConfig = {
  enabled: boolean;           // Enable reconnection (default: true)
  maxRetries: number;         // Maximum retry attempts (default: 5)
  initialDelay: number;       // Initial delay in ms (default: 1000)
  maxDelay: number;           // Maximum delay in ms (default: 30000)
  backoffMultiplier: number;  // Backoff multiplier (default: 2)
};

type TimeoutConfig = {
  connect?: number;    // Connection timeout in ms (default: 10000)
  message?: number;    // Message timeout in ms (default: 30000)
  operation?: number;  // Operation timeout in ms (default: 120000)
};

type LogConfig = {
  level: "debug" | "info" | "warn" | "error" | "silent";  // Log level (default: "info")
  prefix?: string;  // Log prefix (default: "[ComfyClient]")
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
- `ComfyReconnectError` - Reconnection failures (max retries reached)

All errors are part of the `ComfyClientError` union type.

## Adapters

The package provides a universal adapter that works in both Bun and Browser environments:

```typescript
import { UniversalWebSocketAdapter } from '@cj/comfy-client/adapters/universal';

const adapter = new UniversalWebSocketAdapter();
```

For testing, a mock adapter is also available:

```typescript
import { MockWebSocketAdapter } from '@cj/comfy-client/adapters/mock';

const adapter = new MockWebSocketAdapter();
```

## Utilities

The package exports utility functions for working with ComfyUI messages:

```typescript
import {
  extractImagesFromMessage,
  extractImagesFromExecuted,
  extractImagesFromUi,
  extractImagesFromBinary,
  convertBinaryToDataUrl,
  normalizeImageDataUrl
} from '@cj/comfy-client/utils';
```

## Examples

See the test files for complete usage examples:
- `tests/bun-tests.test.ts` - Server-side usage with Bun
- `AUTO_CONNECT_README.md` - Auto-connect and tab focus reconnection features

## License

MIT
