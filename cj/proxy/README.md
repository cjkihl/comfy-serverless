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
- **Circuit Breaker**: Graceful degradation when ComfyUI is down
- **Prompt Queuing**: Queues prompts during downtime (max 100 per user)
- **Session Management**: Automatic cleanup of inactive sessions
- **Monitoring**: Metrics endpoint with secret authentication

## Security

⚠️ **IMPORTANT PRODUCTION SECURITY SETTINGS:**

1. **NO_VERIFY Flag**: The `NO_VERIFY` environment variable disables JWT verification. **NEVER** set this to `true` in production environments. This flag exists only for development and testing purposes.

2. **METRICS_SECRET**: The default value "123456" is insecure. **MUST** be changed to a strong, random secret in production to protect your metrics endpoint.

3. **JWT Configuration**: Always configure `JWT_JWKS_URL`, `JWT_ISSUERS`, and `JWT_AUDIENCES` in production to enforce proper token validation.

## Environment Variables

### Core Configuration
- `PORT`: Proxy port (default: 8190)
- `COMFY_URL`: **ComfyUI backend URL** - ⚠️ **REQUIRED** - Set this to your remote ComfyUI server (e.g., `http://remote-server:8188`)

### Connections
- `MAX_CONNECTIONS_PER_USER`: Maximum concurrent connections per user (default: 1)

### Session Management
- `SESSION_TIMEOUT_MS`: Session timeout in milliseconds (default: 1800000 = 30 minutes)
- `CLEANUP_INTERVAL_MS`: Session cleanup interval in milliseconds (default: 300000 = 5 minutes)

### Error Handling
- `MAX_QUEUED_PROMPTS_PER_USER`: Maximum queued prompts per user during downtime (default: 100)

### Monitoring
- `METRICS_SECRET`: Secret key for metrics endpoint authentication (default: "123456") ⚠️ **CHANGE IN PRODUCTION**
- `NO_VERIFY`: Disable JWT verification (default: false) ⚠️ **NEVER TRUE IN PRODUCTION**

### JWT Configuration
- `JWT_ISSUERS`: Comma-separated issuer allowlist
- `JWT_AUDIENCES`: Comma-separated audience allowlist  
- `JWT_ALG_ALLOWLIST`: Comma-separated algorithm allowlist (default: EdDSA)
- `JWT_JWKS_URL`: JWKS endpoint URL

## Usage

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8190/ws', {
  headers: {
    'Authorization': 'Bearer <jwt-token>'
  }
});

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
- **Failover**: Automatic reconnection on connection drops

### Session Management
- **Timeout**: Sessions expire after 30 minutes of inactivity
- **Cleanup**: Automatic cleanup every 5 minutes
- **Limits**: 1 concurrent connection per user by default

### Error Handling
- **Circuit Breaker**: Opens after 5 consecutive failures
- **Queuing**: Up to 100 prompts per user during downtime
- **Recovery**: Automatic retry when ComfyUI comes back online

### Performance
- **Memory**: ~1MB per 1000 active sessions
- **CPU**: Minimal overhead, mostly I/O bound
- **Network**: Shared WebSocket connections reduce ComfyUI load

## Development

⚠️ **IMPORTANT**: Before starting the proxy, you **MUST** set the `COMFY_URL` environment variable:

```bash
cd proxy

# Set the ComfyUI server URL (REQUIRED)
export COMFY_URL=http://your-remote-comfyui-server:8188

# Install dependencies
bun install

# Start the proxy
bun run dev
```

The proxy will connect to the ComfyUI server specified in `COMFY_URL` and listen on port 8190 (configurable via `PORT`).

## Docker

The proxy is automatically included in the Docker container and managed by Supervisor alongside ComfyUI.

## Message Types

**Outbound (Client → Proxy):**
- `submit_prompt` - Submit a new workflow
- `ping` - Keep-alive ping

**Inbound (Proxy → Client):**
- `prompt_accepted` - Workflow was accepted and queued
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
