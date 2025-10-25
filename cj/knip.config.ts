import type { KnipConfig } from 'knip';

const config: KnipConfig = {
  workspaces: {
    client: {
      entry: ['src/index.ts'],
      project: ['src/**/*.{ts,tsx,js,jsx}', '!src/**/*.test.*', '!src/**/*.spec.*', '!tests/**'],
      ignoreDependencies: ['bun-types'],
    },
    proxy: {
      entry: ['src/server.ts'],
      project: ['src/**/*.{ts,tsx,js,jsx}', '!src/**/*.test.*', '!src/**/*.spec.*', '!tests/**'],
      ignoreDependencies: ['@types/jsonwebtoken'],
    },
  },
  ignore: [
    '**/node_modules/**',
    '**/dist/**',
    '**/build/**',
    '**/.git/**',
    '**/coverage/**',
    '**/*.d.ts',
    '**/examples/**',
    '**/tests/**',
  ],
  ignoreExportsUsedInFile: true,
  ignoreDependencies: [
    '@biomejs/biome',
    '@cjkihl/create-exports',
    '@cjkihl/tsconfig',
    '@cjkihl/with-env',
    '@manypkg/cli',
    'sort-package-json',
    '@types/bun',
    '@types/node',
    'typescript',
  ],
};

export default config;
