import { test, expect } from "bun:test";
import { createApp } from "../src/http/app.ts";
import type { RepoYetiConfig } from "../src/config.ts";
import type { AiPassClient } from "../src/aipass.ts";

// Local mode (no OIDC) → /api/* is not gated, so we can exercise the AI routes directly.
const localCfg = (): RepoYetiConfig => ({ roots: [], port: 7171, maxDepth: 6, maxRepos: 200 });

const post = (app: ReturnType<typeof createApp>, path: string, body: unknown) =>
  app.request(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

const fakeAiPass = (overrides: Partial<AiPassClient> = {}): AiPassClient => ({
  beginAuthorization: async () =>
    "https://aipass.one/oauth2/authorize?client_id=protected-test-client",
  cancelAuthorization: () => {},
  completeAuthorization: async () => ({
    models: [
      { id: "live-wallet-model", label: "Live Wallet Model" },
      { id: "another-wallet-model", label: "Another Wallet Model" },
    ],
  }),
  isConnected: async () => false,
  listModels: async () => [{ id: "live-wallet-model", label: "Live Wallet Model" }],
  streamCompletion: async () => "feat: use the connected wallet",
  disconnect: async () => ({ revoked: true }),
  ...overrides,
});

test("GET /api/ai/settings starts with no configured AI provider", async () => {
  const res = await createApp(localCfg()).request("/api/ai/settings");
  expect(res.status).toBe(200);
  const j = await res.json();
  expect(j.defaultProvider).toBeNull();
  expect(j.providers.groq).toBeUndefined();
  expect(j.style).toBe("conventional");
  expect(JSON.stringify(j)).not.toContain("apiKey");
});

test("connect with an empty key is rejected before any network call", async () => {
  const res = await post(createApp(localCfg()), "/api/ai/providers/openai/connect", {});
  expect(res.status).toBe(400);
  expect((await res.json()).code).toBe("NO_KEY");
});

test("connect to an unknown provider → 404 BAD_PROVIDER", async () => {
  const res = await post(createApp(localCfg()), "/api/ai/providers/bogus/connect", { apiKey: "x" });
  expect(res.status).toBe(404);
  expect((await res.json()).code).toBe("BAD_PROVIDER");
});

test("commit-message refuses to run until an AI provider is configured", async () => {
  const res = await post(createApp(localCfg()), "/api/repos/whatever/commit-message", {});
  expect(res.status).toBe(400);
  expect((await res.json()).code).toBe("NO_AI_PROVIDER");
});

test("setting a default provider that has no key is refused", async () => {
  const res = await createApp(localCfg()).request("/api/ai/settings", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ defaultProvider: "openai" }),
  });
  expect(res.status).toBe(400);
  expect((await res.json()).code).toBe("NOT_CONFIGURED");
});

test("AI Pass is catalogued as an account connection, not an API-key provider", async () => {
  const res = await createApp(localCfg(), { aiPass: fakeAiPass() }).request("/api/ai/catalog");
  const body = (await res.json()) as {
    catalog: Array<Record<string, unknown>>;
  };
  const entry = body.catalog.find((candidate) => candidate.id === "aipass");

  expect(entry).toMatchObject({
    id: "aipass",
    label: "AI Pass",
    accountConnection: true,
  });
  expect(entry).not.toHaveProperty("keyPlaceholder");

  const keyRoute = await post(
    createApp(localCfg(), { aiPass: fakeAiPass() }),
    "/api/ai/providers/aipass/connect",
    { apiKey: "must-not-be-accepted" },
  );
  expect(keyRoute.status).toBe(400);
  expect((await keyRoute.json()).message).toContain("Connect AI Pass");
});

test("Connect AI Pass starts a server-owned authorization redirect", async () => {
  let origin = "";
  const aiPass = fakeAiPass({
    beginAuthorization: async (value) => {
      origin = value;
      return "https://aipass.one/oauth2/authorize?state=opaque";
    },
  });
  const res = await createApp(localCfg(), { aiPass }).request(
    "http://127.0.0.1:7171/api/ai/aipass/connect",
  );

  expect(res.status).toBe(302);
  expect(res.headers.get("location")).toBe(
    "https://aipass.one/oauth2/authorize?state=opaque",
  );
  expect(origin).toBe("http://127.0.0.1:7171");
});

test("AI Pass callback stores only connection/model metadata in app config", async () => {
  const cfg = localCfg();
  const aiPass = fakeAiPass();
  const res = await createApp(cfg, { aiPass }).request(
    "http://127.0.0.1:7171/aipass/oauth/callback?code=one-time&state=opaque",
  );

  expect(res.status).toBe(200);
  expect(res.headers.get("content-type")).toContain("text/html");
  expect(res.headers.get("cache-control")).toBe("no-store");
  expect(res.headers.get("referrer-policy")).toBe("no-referrer");
  expect(res.headers.get("content-security-policy")).toContain("default-src 'none'");
  expect(await res.text()).toContain("AI Pass connected");
  expect(cfg.ai?.providers.aipass).toEqual({
    connected: true,
    model: "live-wallet-model",
  });
  expect(cfg.ai?.defaultProvider).toBe("aipass");
  expect(JSON.stringify(cfg)).not.toContain("one-time");
  expect(JSON.stringify(cfg)).not.toContain("token");

  const settings = await createApp(cfg, { aiPass }).request("/api/ai/settings");
  expect(await settings.json()).toMatchObject({
    providers: { aipass: { configured: true, model: "live-wallet-model" } },
    defaultProvider: "aipass",
  });
});

test("AI Pass authorization errors consume state without attempting a token exchange", async () => {
  let canceledState = "";
  const aiPass = fakeAiPass({
    cancelAuthorization: (state) => {
      canceledState = state;
    },
    completeAuthorization: async () => {
      throw new Error("must not exchange");
    },
  });

  const res = await createApp(localCfg(), { aiPass }).request(
    "http://127.0.0.1:7171/aipass/oauth/callback?error=access_denied&state=opaque",
  );

  expect(res.status).toBe(400);
  expect(canceledState).toBe("opaque");
  expect(res.headers.get("cache-control")).toBe("no-store");
});

test("AI Pass model refresh is live and disconnect revokes before clearing provider state", async () => {
  const cfg: RepoYetiConfig = {
    ...localCfg(),
    ai: {
      providers: { aipass: { connected: true, model: "old-live-model" } },
      defaultProvider: "aipass",
    },
  };
  let disconnected = 0;
  const aiPass = fakeAiPass({
    isConnected: async () => true,
    disconnect: async () => {
      disconnected++;
      return { revoked: false };
    },
  });
  const app = createApp(cfg, { aiPass });

  const listed = await app.request("/api/ai/providers/aipass/models");
  expect(await listed.json()).toEqual({
    ok: true,
    models: [{ id: "live-wallet-model", label: "Live Wallet Model" }],
  });

  const removed = await app.request("/api/ai/providers/aipass", { method: "DELETE" });
  expect(removed.status).toBe(200);
  expect(disconnected).toBe(1);
  expect(cfg.ai?.providers.aipass).toBeUndefined();
  expect(cfg.ai?.defaultProvider).toBeUndefined();
  expect(await removed.json()).toMatchObject({ defaultProvider: null });
});
