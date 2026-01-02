# Contributing to @cj Packages

This guide is specifically for contributing to the `@cj/` packages, which are TypeScript packages isolated from the Python ComfyUI codebase.

## Package Structure

```
cj/
├── comfy-auth/      # Shared JWT authentication utilities
├── comfy-client/    # WebSocket client for ComfyUI
├── comfy-proxy/     # WebSocket proxy server
└── web/             # Browser test suite and development UI
```

## Development Setup

### Prerequisites

- **Bun** (v1.3+): Runtime and package manager
- **Node.js** (v18+): For Playwright tests (web package)
- **TypeScript**: Type checking

### Initial Setup

```bash
# Install dependencies for all packages
cd cj
bun install

# Or install for specific package
cd cj/client
bun install
```

### Environment Variables

Create `.env` files as needed:

**For proxy:**
```bash
cd cj/proxy
# .env
PROXY_COMFY_URL=http://your-comfyui-server:8188
PROXY_COMFY_JWT_SECRET=your-secret-key
LOG_LEVEL=debug
```

**For web tests:**
```bash
cd cj/web
# .env
PROXY_COMFY_JWT_SECRET=your-secret-key
PROXY_URL=ws://localhost:8190/ws
```

## Development Workflow

### 1. Making Changes

1. **Create a branch** from main
2. **Make your changes** following the code style
3. **Run tests** to ensure nothing breaks
4. **Update documentation** if needed
5. **Submit a PR** with a clear description

### 2. Code Style

- **TypeScript**: Use strict mode, prefer type inference where possible
- **Formatting**: Use Biome (configured in `biome.json`)
- **Naming**: 
  - PascalCase for classes/types
  - camelCase for functions/variables
  - UPPER_CASE for constants
- **Imports**: Group imports (external, internal, types)

### 3. Testing

#### Unit Tests

```bash
# Client tests
cd cj/client
bun test

# Proxy tests
cd cj/proxy
bun test
```

#### Integration Tests

```bash
# Run integration tests
cd cj/proxy
bun test tests/integration.test.ts

cd cj/client
bun test tests/integration.test.ts
```

#### E2E Tests

```bash
# Browser tests (requires proxy running)
cd cj/web
bun run test:web

# Bun tests
bun run test:bun

# All tests
bun run test:all
```

### 4. Type Checking

```bash
# Check types
cd cj/<package>
bun run type-check
```

## Package-Specific Guidelines

### @cj/comfy-auth

- **Purpose**: Shared authentication utilities
- **Dependencies**: Minimal (only `jsonwebtoken`)
- **Exports**: Public API only (`index.pub.ts`)
- **Testing**: Unit tests for JWT operations

### @cj/comfy-client

- **Purpose**: Universal WebSocket client
- **Key Files**:
  - `client.ts` - Main client implementation
  - `adapters/` - WebSocket adapter abstractions
  - `errors.ts` - Error types and Logger
  - `types.ts` - Type definitions
- **Testing**: 
  - Unit tests with MockWebSocketAdapter
  - Integration tests with real adapter

### @cj/comfy-proxy

- **Purpose**: WebSocket proxy server
- **Key Files**:
  - `proxy.ts` - Main proxy implementation
  - `sessionManager.ts` - Session lifecycle management
  - `errorHandling.ts` - Circuit breaker and prompt queue
  - `logger.ts` - Logger with correlation IDs
- **Testing**:
  - Unit tests for components
  - Integration tests for flows
  - Health check tests

### @cj/web

- **Purpose**: Browser testing and dev UI
- **Key Files**:
  - `tests/` - Test suites
  - `src/` - React components for testing UI
- **Testing**: Playwright E2E tests

## Adding New Features

### 1. Plan Your Changes

- Check if feature belongs in existing package or needs new package
- Consider impact on isolation from Python code
- Document breaking changes

### 2. Implementation

- Follow existing patterns
- Add tests for new functionality
- Update types and exports
- Add JSDoc comments for public APIs

### 3. Testing

- Add unit tests for new code
- Add integration tests if touching multiple packages
- Update E2E tests if changing user-facing behavior

### 4. Documentation

- Update README.md if adding features
- Add JSDoc comments
- Update ARCHITECTURE.md if changing architecture
- Add examples if introducing new patterns

## Logging Guidelines

All packages use **consola** for logging:

```typescript
import { Logger } from "./logger";

const logger = new Logger("info", "[MyComponent]");

// Use structured logging with objects
logger.info("Operation completed", {
  userId: "user123",
  durationMs: 150,
  correlationId: "abc-123",
});

// Use correlation IDs for request tracing
const sessionLogger = logger.withCorrelationId(correlationId);
sessionLogger.debug("Processing request");
```

**Best Practices**:
- Use appropriate log levels (debug, info, warn, error)
- Include context in log messages (userId, promptId, etc.)
- Use correlation IDs for request tracing
- Avoid logging sensitive data

## Error Handling

Use Result types for explicit error handling:

```typescript
import { Result, err, ok } from "./types";

async function myFunction(): Promise<Result<string>> {
  try {
    const result = await doSomething();
    return ok(result);
  } catch (error) {
    return err(new MyError("Something went wrong", error));
  }
}
```

## Testing Guidelines

### Unit Tests

- Test one thing at a time
- Use descriptive test names
- Mock external dependencies
- Test edge cases and error paths

### Integration Tests

- Test package interactions
- Use real adapters when possible
- Test error propagation
- Verify correlation ID flow

### E2E Tests

- Test complete user flows
- Use real proxy server
- Test in actual browser (Playwright)
- Verify image generation end-to-end

## Pull Request Process

1. **Create PR** with clear title and description
2. **Link issues** if fixing bugs or implementing features
3. **Ensure tests pass** - CI will run tests
4. **Update documentation** if needed
5. **Request review** from maintainers

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Type checking passes
- [ ] No breaking changes (or documented)
- [ ] Logging follows guidelines
- [ ] Error handling uses Result types

## Code Review Guidelines

### For Authors

- Keep PRs focused and small
- Respond to feedback promptly
- Update PR based on comments
- Test changes locally before requesting re-review

### For Reviewers

- Be constructive and specific
- Check test coverage
- Verify isolation from Python code
- Ensure documentation is updated

## Common Patterns

### Creating a New Package

1. Create directory: `cj/my-package/`
2. Add `package.json` with workspace dependencies
3. Add `tsconfig.json` extending shared config
4. Create `index.pub.ts` for public exports
5. Add README.md
6. Add tests

### Adding Shared Utilities

- Consider if utility belongs in existing package
- Extract to shared package if used by multiple packages
- Use workspace protocol for dependencies
- Document usage in README

### Adding New Logger Features

- Extend Logger class in `logger.ts`
- Maintain backward compatibility
- Add tests for new features
- Document in JSDoc comments

## Troubleshooting

### Tests Failing

1. Check environment variables are set
2. Verify proxy server is running (for integration tests)
3. Check ComfyUI server is accessible
4. Review test logs for specific errors

### Type Errors

1. Run `bun run type-check`
2. Check imports are correct
3. Verify types are exported properly
4. Check workspace dependencies are installed

### Build Issues

1. Clear `dist/` and `node_modules/`
2. Run `bun install` again
3. Check for version conflicts
4. Verify Bun version is compatible

## Questions?

- Check existing documentation in package READMEs
- Review ARCHITECTURE.md for system overview
- Look at existing code for patterns
- Ask in team chat or create an issue

## License

All @cj packages are MIT licensed, maintaining compatibility with ComfyUI's license.



