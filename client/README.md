# MindFlow Tutor — Client

React 18 + TypeScript + Vite frontend for MindFlow Tutor. See the [root README](../README.md) for the full setup.

## Scripts

| Command           | Description                                      |
| ----------------- | ------------------------------------------------ |
| `npm run dev`     | Dev server on `http://127.0.0.1:5173` (proxies `/api` to `:8000`) |
| `npm run build`   | Production build into `dist/`                    |
| `npm run lint`    | ESLint                                           |
| `npm test`        | Unit tests (Vitest)                              |
| `npx playwright test` | End-to-end tests in `e2e/` (Playwright)      |

## Structure

```
src/
  components/   feature components (+ ui/ — shadcn/ui primitives)
  contexts/     React contexts (auth)
  hooks/        shared hooks
  lib/          API clients and helpers
  pages/        route-level pages
```
