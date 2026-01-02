# ComfyUI Proxy

A production-ready WebSocket proxy for ComfyUI that provides JWT authentication and graceful degradation for concurrent users, with a dedicated ComfyUI WebSocket per user.

## Architecture

```
Client WebSocket (JWT Auth) → Proxy → ComfyUI HTTP POST /prompt
                                      ↓
Client WebSocket ← Proxy ← ComfyUI WebSocket (Real-time Updates)
```

**Key Features:**
- **JWT Authentication**: Verifies tokens once on WebSocket upgrade
- **Per-user ComfyUI Connection**: Each user gets a dedicated WebSocket to ComfyUI
- **HTTP Prompt Submission**: Forwards prompts via HTTP POST to ComfyUI's `/prompt` endpoint
- **Real-time Updates**: Proxies WebSocket messages for progress, status, and results
- **Automatic Reconnection**: Automatically reconnects to ComfyUI when connection drops
- **Circuit Breaker**: Graceful degradation when ComfyUI is down
- **Prompt Queuing**: Queues prompts during downtime (max 100 per user)
- **Session Management**: Automatic cleanup of inactive sessions
- **Monitoring**: Metrics endpoint with secret authentication

## Security

⚠️ **IMPORTANT PRODUCTION SECURITY SETTINGS:**

1. **METRICS_SECRET**: The default value "123456" is insecure. **MUST** be changed to a strong, random secret in production to protect your metrics endpoint.

2. **JWT Configuration**: Always configure `PROXY_COMFY_JWT_SECRET` in production to enforce proper token validation.

## Environment Variables

### Required Variables
- `PROXY_COMFY_URL`: **REQUIRED** - ComfyUI backend URL (e.g., `http://remote-server:8188`)
- `PROXY_COMFY_JWT_SECRET`: **REQUIRED** - HMAC secret used to sign and verify JWT tokens (HS256 algorithm)

### Core Configuration (Optional)
- `PROXY_PORT`: Proxy port (default: 8190)
- `LOG_LEVEL`: Log level - `"debug" | "info" | "warn" | "error" | "silent"` (default: "info")
  - Use `"debug"` for development, `"info"` or `"warn"` for production

### Connections (Optional)
- `MAX_CONNECTIONS_PER_USER`: Maximum concurrent connections per user (default: 5). When limit is reached, least active session is automatically evicted.

### Reconnection Configuration (Optional)
- `PROXY_COMFY_RECONNECT_ENABLED`: Enable automatic reconnection to ComfyUI (default: true)
  - When enabled, the proxy will automatically attempt to reconnect to ComfyUI if the connection drops
  - Client connections remain open during reconnection attempts
- `PROXY_COMFY_RECONNECT_MAX_RETRIES`: Maximum reconnection attempts (default: 5)
  - After this many failed attempts, the client connection is closed
- `PROXY_COMFY_RECONNECT_INITIAL_DELAY_MS`: Initial reconnection delay in milliseconds (default: 1000 = 1 second)
  - First reconnection attempt happens after this delay
- `PROXY_COMFY_RECONNECT_MAX_DELAY_MS`: Maximum reconnection delay in milliseconds (default: 30000 = 30 seconds)
  - Exponential backoff is capped at this value

### Session Management (Optional)
- `SESSION_TIMEOUT_MS`: Session timeout in milliseconds (default: 1800000 = 30 minutes)
  - Sessions expire after this period of inactivity
- `SESSION_IDLE_EVICTION_MS`: Session idle eviction timeout in milliseconds (default: 300000 = 5 minutes)
  - Sessions idle for this long are evicted when connection limit is reached
- `CLEANUP_INTERVAL_MS`: Session cleanup interval in milliseconds (default: 300000 = 5 minutes)
  - How often to run cleanup of inactive sessions

### Error Handling & Circuit Breaker (Optional)
- `MAX_QUEUED_PROMPTS_PER_USER`: Maximum queued prompts per user during downtime (default: 100)
  - Prompts are queued when ComfyUI is unavailable and processed when it recovers
- `CIRCUIT_BREAKER_THRESHOLD`: Failures before opening circuit (default: 5)
  - After this many consecutive failures, circuit opens to prevent cascading failures
- `CIRCUIT_BREAKER_TIMEOUT_MS`: Circuit breaker timeout in milliseconds (default: 30000 = 30 seconds)
  - How long circuit stays open before attempting half-open state
- `MAX_PROMPT_RETRIES`: Maximum number of retries for a queued prompt (default: 3)
  - Prompts exceeding this are dropped
- `INITIAL_RETRY_DELAY_MS`: Initial retry delay in milliseconds (default: 1000 = 1 second)
  - First retry happens after this delay, then exponential backoff
- `MAX_RETRY_DELAY_MS`: Maximum retry delay in milliseconds (default: 30000 = 30 seconds)
  - Exponential backoff is capped at this value

### Health Check Configuration (Optional)
- `HEALTH_CHECK_TIMEOUT_MS`: Health check timeout in milliseconds (default: 5000 = 5 seconds)
  - Timeout for individual health check requests to ComfyUI
- `HEALTH_CHECK_INTERVAL_MS`: Health check interval in milliseconds (default: 10000 = 10 seconds)
  - How often to perform periodic health checks on ComfyUI

### Request Timeouts (Optional)
- `HTTP_REQUEST_TIMEOUT_MS`: HTTP request timeout in milliseconds (default: 30000 = 30 seconds)
  - Timeout for HTTP POST requests to ComfyUI /prompt endpoint
- `SESSION_READY_TIMEOUT_MS`: Session ready timeout in milliseconds (default: 10000 = 10 seconds)
  - Timeout for waiting for ComfyUI session to be ready (sid received)
- `CONNECTION_TIMEOUT_MS`: Connection timeout in milliseconds (default: 10000 = 10 seconds)
  - Timeout for establishing WebSocket connection to ComfyUI

### Monitoring (Optional)
- `METRICS_SECRET`: Secret key for metrics endpoint authentication (default: "123456") ⚠️ **CHANGE IN PRODUCTION**

**Note**: Tokens are verified using HMAC secret-based verification with HS256 algorithm (hardcoded). The `PROXY_COMFY_JWT_SECRET` must be shared between the token issuer and the proxy. Tokens can be from any issuer (no issuer/audience validation). The `sub` claim is required and used as the userId.

### Configuration Examples

**Minimal Configuration (Required Only):**
```bash
export PROXY_COMFY_URL=http://remote-server:8188
export PROXY_COMFY_JWT_SECRET=your-secret-key-here
```

**Production Configuration:**
```bash
export PROXY_COMFY_URL=http://remote-server:8188
export PROXY_COMFY_JWT_SECRET=your-strong-secret-key
export METRICS_SECRET=your-metrics-secret
export LOG_LEVEL=info
export MAX_CONNECTIONS_PER_USER=10
export SESSION_TIMEOUT_MS=3600000  # 1 hour
```

**Development Configuration (Verbose Logging):**
```bash
export PROXY_COMFY_URL=http://localhost:8188
export PROXY_COMFY_JWT_SECRET=dev-secret
export LOG_LEVEL=debug
export HEALTH_CHECK_INTERVAL_MS=5000  # Check more frequently
```

## Usage

### WebSocket Connection
```javascript
// Note: WebSockets don't support custom headers, so pass token as query parameter
const ws = new WebSocket('ws://localhost:8190/ws?token=<jwt-token>');

// Submit prompt over WebSocket
ws.send(JSON.stringify({
  type: 'submit_prompt',
  data: {
    prompt: { /* ComfyUI workflow */ },
    prompt_id: 'optional-prompt-id',
    extra_data: { /* optional metadata */ },
    partial_execution_targets: ['node1', 'node2'] // optional
  }
}));

// Listen for responses
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  switch (message.type) {
    case 'prompt_accepted':
      console.log('Prompt accepted:', message.data);
      break;
    case 'status':
      console.log('Queue status:', message.data);
      break;
    case 'executing':
      console.log('Executing node:', message.data.node);
      break;
    case 'progress':
      console.log('Progress:', message.data);
      break;
    case 'executed':
      console.log('Node completed:', message.data);
      break;
    case 'error':
      console.error('Error:', message.data);
      break;
  }
};
```

## Monitoring Endpoints

### Health Checks
- `GET /health` - Basic health check
- `GET /live` - Process is alive
- `GET /ready` - Ready to accept traffic (checks ComfyUI connectivity)

### Metrics (Protected)
- `GET /metrics` - Basic metrics
- `GET /metrics?detailed=true` - Detailed metrics with per-session info

**Authentication**: Include `Authorization: Bearer <METRICS_SECRET>` header

**Example Response**:
```json
{
  "active_sessions": 42,
  "active_connections": 8,
  "uptime_seconds": 3600,
  "memory_usage": {
    "rss": 123456789,
    "heapTotal": 98765432,
    "heapUsed": 87654321,
    "external": 1234567
  },
  "circuit_breaker_state": "closed",
  "queued_prompts": {
    "user123": 5,
    "user456": 2
  }
}
```

## Scaling Considerations

### Connections
- **Per-user Connection**: Each user maintains a dedicated WebSocket to ComfyUI
- **Automatic Reconnection**: When ComfyUI connection drops, proxy automatically attempts to reconnect with exponential backoff
  - Client connection remains open during reconnection attempts
  - Session state is preserved and restored after successful reconnection
  - Clients receive a `reconnected` message when reconnection succeeds
  - After maximum retries, client connection is closed with an error

### Session Management
- **Timeout**: Sessions expire after 30 minutes of inactivity
- **Cleanup**: Automatic cleanup every 5 minutes
- **Limits**: 5 concurrent connections per user by default
- **Eviction**: When limit is reached, least active session is automatically evicted to make room for new connections

### Error Handling
- **Circuit Breaker**: Opens after configured number of consecutive failures (default: 5)
- **Queuing**: Up to configured number of prompts per user during downtime (default: 100)
- **Recovery**: Automatic retry with exponential backoff when ComfyUI comes back online
- **Retry Limits**: Configurable max retries per prompt (default: 3)

### Performance
- **Memory**: ~1MB per 1000 active sessions
- **CPU**: Minimal overhead, mostly I/O bound
- **Network**: Shared WebSocket connections reduce ComfyUI load

## Development

⚠️ **IMPORTANT**: Before starting the proxy, you **MUST** set the `PROXY_COMFY_URL` environment variable. **ComfyUI MUST be running on a remote server** - do not attempt to run ComfyUI locally for production/testing:

```bash
cd proxy

# Set the ComfyUI server URL (REQUIRED - must be remote server)
export PROXY_COMFY_URL=http://your-remote-comfyui-server:8188
export PROXY_COMFY_JWT_SECRET=your-jwt-secret

# Or create a .env file with these variables
# The proxy uses with-env to automatically load .env files

# Install dependencies
bun install

# Start the proxy (with-env automatically loads .env files)
bun run dev
```

The proxy uses `with-env` to automatically load environment variables from `.env` files. The `dev` and `start` scripts are wrapped with `with-env`, which searches for `.env` files starting from the current directory and walking up the directory tree.

The proxy will connect to the ComfyUI server specified in `PROXY_COMFY_URL` and listen on port 8190 (configurable via `PROXY_PORT`).

## Docker

The proxy is automatically included in the Docker container and managed by Supervisor alongside ComfyUI.

## Message Types

**Outbound (Client → Proxy):**
- `submit_prompt` - Submit a new workflow
- `ping` - Keep-alive ping

**Inbound (Proxy → Client):**
- `prompt_accepted` - Workflow was accepted and queued
- `reconnected` - ComfyUI connection restored after reconnection
- `status` - Queue status updates
- `executing` - Currently executing node
- `progress` - Progress updates within a node
- `executed` - Node execution completed
- `error` - Error occurred

## Troubleshooting

### Common Issues

1. **"Maximum connections per user exceeded"**
   - User has too many concurrent connections
   - Check `MAX_CONNECTIONS_PER_USER` setting

2. **"ComfyUI is temporarily unavailable"**
   - Circuit breaker is open due to ComfyUI failures
   - Check ComfyUI health and logs
   - Prompts are queued and will be processed when ComfyUI recovers

3. **"Queue full"**
   - User has too many queued prompts
   - Check `MAX_QUEUED_PROMPTS_PER_USER` setting
   - Consider increasing limit or improving ComfyUI reliability

4. **Connection issues**
   - Monitor `/metrics` endpoint for active sessions and circuit breaker state
   - Ensure ComfyUI is reachable

### Monitoring

Use the `/metrics` endpoint to monitor:
- Active sessions and connections
- Circuit breaker state
- Queued prompts per user
- Memory usage and uptime
