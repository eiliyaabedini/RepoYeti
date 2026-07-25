import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  createAiPassClient,
  parseAiPassModels,
  type AiPassSecretStore,
  type AiPassTokenBundle,
} from "../src/aipass.ts";
import { CONFIG_DIR, saveConfig, type RepoYetiConfig } from "../src/config.ts";

const METADATA = {
  issuer: "https://aipass.one",
  authorization_endpoint: "https://aipass.one/oauth2/authorize",
  token_endpoint: "https://aipass.one/oauth2/token",
  userinfo_endpoint: "https://aipass.one/oauth2/userinfo",
  revocation_endpoint: "https://aipass.one/oauth2/revoke",
  code_challenge_methods_supported: ["S256", "plain"],
  token_endpoint_auth_methods_supported: ["none"],
};

class MemoryStore implements AiPassSecretStore {
  value: string | null = null;
  writes: string[] = [];
  clears = 0;
  allowWrite = true;

  async get(): Promise<string | null> {
    return this.value;
  }

  async set(value: string): Promise<boolean> {
    this.writes.push(value);
    if (!this.allowWrite) return false;
    this.value = value;
    return true;
  }

  async delete(): Promise<boolean> {
    this.clears++;
    this.value = null;
    return true;
  }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function sse(parts: string[], status = 200): Response {
  return new Response(parts.join(""), {
    status,
    headers: { "content-type": "text/event-stream" },
  });
}

test("model discovery accepts OpenAI lists and legacy string arrays without fixed model ids", () => {
  expect(
    parseAiPassModels({
      object: "list",
      data: [
        { id: "wallet-chat-z", name: "Wallet Chat Z", methods: ["chat_completions"] },
        { id: "wallet-audio", name: "Wallet Audio", methods: ["audio_speech"] },
      ],
    }),
  ).toEqual([{ id: "wallet-chat-z", label: "Wallet Chat Z" }]);

  expect(parseAiPassModels(["future-chat-b", "future-chat-a"])).toEqual([
    { id: "future-chat-a", label: "future-chat-a" },
    { id: "future-chat-b", label: "future-chat-b" },
  ]);
});

test("authorization uses discovered endpoints, strong state, and PKCE S256 without a client secret", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store: new MemoryStore(),
    fetchImpl: async (input, init) => {
      calls.push({ url: String(input), init });
      return json(METADATA);
    },
  });

  const location = await client.beginAuthorization("https://app.example.test");
  const url = new URL(location);

  expect(url.origin + url.pathname).toBe(METADATA.authorization_endpoint);
  expect(url.searchParams.get("response_type")).toBe("code");
  expect(url.searchParams.get("code_challenge_method")).toBe("S256");
  expect(url.searchParams.get("code_challenge")).toMatch(/^[A-Za-z0-9_-]{43}$/);
  expect(url.searchParams.get("state")).toMatch(/^[A-Za-z0-9_-]{43}$/);
  expect(url.searchParams.get("scope")).toBe("api:access profile:read");
  expect(url.searchParams.has("client_secret")).toBe(false);
  expect(url.searchParams.get("redirect_uri")).toBe(
    "https://app.example.test/aipass/oauth/callback",
  );
  expect(calls).toHaveLength(1);
});

test("callback validates userinfo and models before atomically storing a token bundle", async () => {
  const store = new MemoryStore();
  let tokenBody: Record<string, unknown> | null = null;
  const redirectModes: Array<RequestRedirect | undefined> = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    now: () => 1_000,
    fetchImpl: async (input, init) => {
      redirectModes.push(init?.redirect);
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        tokenBody = JSON.parse(String(init?.body));
        return json({
          access_token: "access-sensitive",
          refresh_token: "refresh-sensitive",
          expires_in: 3600,
          token_type: "Bearer",
        });
      }
      if (url === METADATA.userinfo_endpoint) return json({ sub: "account-1" });
      if (url.includes("/oauth2/v1/models")) {
        return json({
          object: "list",
          data: [{ id: "live-chat", name: "Live Chat", methods: ["chat_completions"] }],
        });
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  const authorization = new URL(await client.beginAuthorization("https://app.example.test"));
  const result = await client.completeAuthorization(
    "one-time-code",
    authorization.searchParams.get("state")!,
  );

  expect(tokenBody).toMatchObject({
    grantType: "authorization_code",
    code: "one-time-code",
    clientId: "protected-test-client",
    redirectUri: "https://app.example.test/aipass/oauth/callback",
  });
  expect(tokenBody).not.toHaveProperty("clientSecret");
  expect(store.writes).toHaveLength(1);
  expect(JSON.parse(store.writes[0]!)).toMatchObject({
    accessToken: "access-sensitive",
    refreshToken: "refresh-sensitive",
    expiresAt: 3_601_000,
  });
  expect(result).toEqual({
    models: [{ id: "live-chat", label: "Live Chat" }],
  });
  expect(redirectModes).toEqual(["error", "error", "error", "error"]);
  expect(JSON.stringify(result)).not.toContain("sensitive");

  await expect(
    client.completeAuthorization("replay", authorization.searchParams.get("state")!),
  ).rejects.toThrow("expired");
});

test("callback fails closed when native secret storage cannot persist the tokens", async () => {
  const store = new MemoryStore();
  store.allowWrite = false;
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    fetchImpl: async (input) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        return json({ access_token: "access-sensitive", refresh_token: "refresh-sensitive" });
      }
      if (url === METADATA.userinfo_endpoint) return json({ sub: "account-1" });
      if (url.includes("/oauth2/v1/models")) return json(["live-chat"]);
      if (url === METADATA.revocation_endpoint) return new Response(null, { status: 200 });
      throw new Error(`unexpected request: ${url}`);
    },
  });
  const authorization = new URL(await client.beginAuthorization("http://127.0.0.1:7171"));

  await expect(
    client.completeAuthorization(
      "one-time-code",
      authorization.searchParams.get("state")!,
    ),
  ).rejects.toThrow("secure storage");
  expect(store.value).toBeNull();
});

test("callback revokes both halves of an unpersisted grant when verification fails", async () => {
  const store = new MemoryStore();
  const revoked: Array<{ token: string; hint: string }> = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        return json({
          access_token: "unpersisted-access",
          refresh_token: "unpersisted-refresh",
        });
      }
      if (url === METADATA.userinfo_endpoint) {
        return json({ error: "temporarily unavailable" }, 503);
      }
      if (url === METADATA.revocation_endpoint) {
        const body = new URLSearchParams(String(init?.body));
        revoked.push({
          token: body.get("token") ?? "",
          hint: body.get("token_type_hint") ?? "",
        });
        return new Response(null, { status: 200 });
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });
  const authorization = new URL(await client.beginAuthorization("https://app.example.test"));

  await expect(
    client.completeAuthorization(
      "one-time-code",
      authorization.searchParams.get("state")!,
    ),
  ).rejects.toThrow("account verification");

  expect(store.writes).toHaveLength(0);
  expect(store.value).toBeNull();
  expect(revoked).toEqual([
    { token: "unpersisted-refresh", hint: "refresh_token" },
    { token: "unpersisted-access", hint: "access_token" },
  ]);
});

test("callback rejects invalid userinfo before storing the grant", async () => {
  const store = new MemoryStore();
  const revokedHints: string[] = [];
  let modelRequests = 0;
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        return json({
          access_token: "unpersisted-access",
          refresh_token: "unpersisted-refresh",
        });
      }
      if (url === METADATA.userinfo_endpoint) return json({});
      if (url.includes("/oauth2/v1/models")) {
        modelRequests++;
        return json(["live-chat"]);
      }
      if (url === METADATA.revocation_endpoint) {
        revokedHints.push(
          new URLSearchParams(String(init?.body)).get("token_type_hint") ?? "",
        );
        return new Response(null, { status: 200 });
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });
  const authorization = new URL(await client.beginAuthorization("https://app.example.test"));

  await expect(
    client.completeAuthorization(
      "one-time-code",
      authorization.searchParams.get("state")!,
    ),
  ).rejects.toThrow("invalid account information");

  expect(modelRequests).toBe(0);
  expect(store.writes).toHaveLength(0);
  expect(revokedHints).toEqual(["refresh_token", "access_token"]);
});

test("callback rejects an unrecognized live-model envelope before storing the grant", async () => {
  const store = new MemoryStore();
  const revokedHints: string[] = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        return json({
          access_token: "unpersisted-access",
          refresh_token: "unpersisted-refresh",
        });
      }
      if (url === METADATA.userinfo_endpoint) return json({ sub: "account-1" });
      if (url.includes("/oauth2/v1/models")) {
        return json({
          object: "unexpected",
          data: [{ id: "not-a-validated-model", methods: ["chat_completions"] }],
        });
      }
      if (url === METADATA.revocation_endpoint) {
        revokedHints.push(
          new URLSearchParams(String(init?.body)).get("token_type_hint") ?? "",
        );
        return new Response(null, { status: 200 });
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });
  const authorization = new URL(await client.beginAuthorization("https://app.example.test"));

  await expect(
    client.completeAuthorization(
      "one-time-code",
      authorization.searchParams.get("state")!,
    ),
  ).rejects.toThrow("model discovery response");

  expect(store.writes).toHaveLength(0);
  expect(revokedHints).toEqual(["refresh_token", "access_token"]);
});

test("callback refuses a non-refreshable grant and revokes its access token", async () => {
  const store = new MemoryStore();
  const revokedHints: string[] = [];
  let userinfoRequests = 0;
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        return json({ access_token: "access-without-refresh", expires_in: 600 });
      }
      if (url === METADATA.userinfo_endpoint) {
        userinfoRequests++;
        return json({ sub: "account-1" });
      }
      if (url === METADATA.revocation_endpoint) {
        revokedHints.push(
          new URLSearchParams(String(init?.body)).get("token_type_hint") ?? "",
        );
        return new Response(null, { status: 200 });
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });
  const authorization = new URL(await client.beginAuthorization("https://app.example.test"));

  await expect(
    client.completeAuthorization(
      "one-time-code",
      authorization.searchParams.get("state")!,
    ),
  ).rejects.toThrow("refresh token");

  expect(userinfoRequests).toBe(0);
  expect(store.writes).toHaveLength(0);
  expect(revokedHints).toEqual(["access_token"]);
});

test("refresh rotation is one atomic bundle write before the retried wallet request", async () => {
  const store = new MemoryStore();
  const old: AiPassTokenBundle = {
    accessToken: "expired-access",
    refreshToken: "old-refresh",
    expiresAt: 1,
  };
  store.value = JSON.stringify(old);
  const order: string[] = [];
  const redirectModes: Array<RequestRedirect | undefined> = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store: {
      get: () => store.get(),
      set: async (value) => {
        order.push("persist-rotation");
        return store.set(value);
      },
      delete: () => store.delete(),
    },
    now: () => 100_000,
    fetchImpl: async (input, init) => {
      redirectModes.push(init?.redirect);
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        expect(JSON.parse(String(init?.body))).toMatchObject({
          grantType: "refresh_token",
          refreshToken: "old-refresh",
          clientId: "protected-test-client",
        });
        return json({
          access_token: "rotated-access",
          refresh_token: "rotated-refresh",
          expires_in: 600,
        });
      }
      if (url.endsWith("/oauth2/v1/chat/completions")) {
        order.push("wallet-request");
        expect(init?.headers).toMatchObject({
          authorization: "Bearer rotated-access",
          accept: "text/event-stream",
        });
        expect(JSON.parse(String(init?.body))).toMatchObject({ stream: true });
        return sse([
          'data: {"choices":[{"delta":{"content":"feat: "}}]}\n\n',
          'data: {"choices":[{"delta":{"content":"rotate safely"}}]}\n\n',
          "data: [DONE]\n\n",
        ]);
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  const text = await client.streamCompletion({ model: "live-chat", messages: [] });

  expect(text).toBe("feat: rotate safely");
  expect(order).toEqual(["persist-rotation", "wallet-request"]);
  expect(redirectModes).toEqual(["error", "error", "error"]);
  expect(store.writes).toHaveLength(1);
  expect(JSON.parse(store.writes[0]!)).toMatchObject({
    accessToken: "rotated-access",
    refreshToken: "rotated-refresh",
  });
});

test("refresh retries one explicit AI Pass rotation conflict without duplicating wallet work", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "expired-access",
    refreshToken: "old-refresh",
    expiresAt: 1,
  } satisfies AiPassTokenBundle);
  let refreshRequests = 0;
  let walletRequests = 0;
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    now: () => 100_000,
    fetchImpl: async (input) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        refreshRequests++;
        if (refreshRequests === 1) {
          return new Response(JSON.stringify({ error: "rotation in progress" }), {
            status: 503,
            headers: {
              "content-type": "application/json",
              "retry-after": "0",
            },
          });
        }
        return json({
          access_token: "rotated-access",
          refresh_token: "rotated-refresh",
          expires_in: 600,
        });
      }
      if (url.endsWith("/oauth2/v1/chat/completions")) {
        walletRequests++;
        return sse([
          'data: {"choices":[{"delta":{"content":"feat: retry refresh"}}]}\n\n',
          "data: [DONE]\n\n",
        ]);
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  expect(
    await client.streamCompletion({ model: "live-chat", messages: [] }),
  ).toBe("feat: retry refresh");
  expect(refreshRequests).toBe(2);
  expect(walletRequests).toBe(1);
});

test("refresh rotation clears stale credentials when the native-store replacement fails", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "expired-access",
    refreshToken: "old-refresh",
    expiresAt: 1,
  } satisfies AiPassTokenBundle);
  store.allowWrite = false;
  let walletRequests = 0;
  const revoked: Array<{ token: string; hint: string }> = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    now: () => 100_000,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        return json({
          access_token: "rotated-access",
          refresh_token: "rotated-refresh",
          expires_in: 600,
        });
      }
      if (url.endsWith("/oauth2/v1/chat/completions")) {
        walletRequests++;
        return sse(["data: [DONE]\n\n"]);
      }
      if (url === METADATA.revocation_endpoint) {
        const body = new URLSearchParams(String(init?.body));
        revoked.push({
          token: body.get("token") ?? "",
          hint: body.get("token_type_hint") ?? "",
        });
        return new Response(null, { status: 200 });
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  await expect(
    client.streamCompletion({ model: "live-chat", messages: [] }),
  ).rejects.toThrow("secure storage");
  expect(walletRequests).toBe(0);
  expect(store.value).toBeNull();
  expect(store.clears).toBe(1);
  expect(revoked).toEqual([
    { token: "rotated-refresh", hint: "refresh_token" },
    { token: "rotated-access", hint: "access_token" },
    { token: "old-refresh", hint: "refresh_token" },
    { token: "expired-access", hint: "access_token" },
  ]);
});

test("refresh fails closed when AI Pass omits the rotated refresh token", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "expired-access",
    refreshToken: "old-refresh",
    expiresAt: 1,
  } satisfies AiPassTokenBundle);
  let walletRequests = 0;
  const revoked: Array<{ token: string; hint: string }> = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    now: () => 100_000,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        return json({ access_token: "rotated-access", expires_in: 600 });
      }
      if (url.endsWith("/oauth2/v1/chat/completions")) {
        walletRequests++;
        return sse(["data: [DONE]\n\n"]);
      }
      if (url === METADATA.revocation_endpoint) {
        const body = new URLSearchParams(String(init?.body));
        revoked.push({
          token: body.get("token") ?? "",
          hint: body.get("token_type_hint") ?? "",
        });
        return new Response(null, { status: 200 });
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  await expect(
    client.streamCompletion({ model: "live-chat", messages: [] }),
  ).rejects.toThrow("refresh token");
  expect(walletRequests).toBe(0);
  expect(store.value).toBeNull();
  expect(revoked).toEqual([
    { token: "rotated-access", hint: "access_token" },
    { token: "old-refresh", hint: "refresh_token" },
    { token: "expired-access", hint: "access_token" },
  ]);
});

test("a rejected fresh token refreshes once and retries the wallet request once", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "rejected-access",
    refreshToken: "current-refresh",
    expiresAt: 200_000,
  } satisfies AiPassTokenBundle);
  let refreshRequests = 0;
  let chatRequests = 0;
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    now: () => 100_000,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        refreshRequests++;
        return json({
          access_token: "rotated-access",
          refresh_token: "rotated-refresh",
          expires_in: 600,
        });
      }
      if (url.endsWith("/oauth2/v1/chat/completions")) {
        chatRequests++;
        if (init?.headers && (init.headers as Record<string, string>).authorization === "Bearer rejected-access") {
          return json({ error: "expired" }, 401);
        }
        return sse([
          'data: {"choices":[{"delta":{"content":"feat: refreshed once"}}]}\n\n',
          "data: [DONE]\n\n",
        ]);
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  expect(
    await client.streamCompletion({ model: "live-chat", messages: [] }),
  ).toBe("feat: refreshed once");
  expect(refreshRequests).toBe(1);
  expect(chatRequests).toBe(2);
});

test("a proactively refreshed token rejected by AI Pass is cleared without rotating twice", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "expired-access",
    refreshToken: "old-refresh",
    expiresAt: 1,
  } satisfies AiPassTokenBundle);
  let refreshRequests = 0;
  let chatRequests = 0;
  const revokedHints: string[] = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    now: () => 100_000,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        refreshRequests++;
        return json({
          access_token: `rotated-access-${refreshRequests}`,
          refresh_token: `rotated-refresh-${refreshRequests}`,
          expires_in: 600,
        });
      }
      if (url.endsWith("/oauth2/v1/chat/completions")) {
        chatRequests++;
        return json({ error: "rejected" }, 401);
      }
      if (url === METADATA.revocation_endpoint) {
        revokedHints.push(
          new URLSearchParams(String(init?.body)).get("token_type_hint") ?? "",
        );
        return new Response(null, { status: 200 });
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  await expect(
    client.streamCompletion({ model: "live-chat", messages: [] }),
  ).rejects.toThrow("authorization expired");
  expect(refreshRequests).toBe(1);
  expect(chatRequests).toBe(1);
  expect(store.value).toBeNull();
  expect(revokedHints).toEqual(["refresh_token", "access_token"]);
});

test("an access-only stored bundle cannot start wallet work", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "orphaned-access",
    expiresAt: 200_000,
  } satisfies AiPassTokenBundle);
  let walletRequests = 0;
  const revokedHints: string[] = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    now: () => 100_000,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.revocation_endpoint) {
        revokedHints.push(
          new URLSearchParams(String(init?.body)).get("token_type_hint") ?? "",
        );
        return new Response(null, { status: 200 });
      }
      if (url.endsWith("/oauth2/v1/chat/completions")) {
        walletRequests++;
        return sse([
          'data: {"choices":[{"delta":{"content":"must not run"}}]}\n\n',
          "data: [DONE]\n\n",
        ]);
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  await expect(
    client.streamCompletion({ model: "live-chat", messages: [] }),
  ).rejects.toThrow("authorization expired");
  expect(walletRequests).toBe(0);
  expect(store.value).toBeNull();
  expect(revokedHints).toEqual(["access_token"]);
});

test("disconnect cannot be undone by an in-flight refresh or followed by wallet work", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "expired-access",
    refreshToken: "old-refresh",
    expiresAt: 1,
  } satisfies AiPassTokenBundle);
  let resolveRefresh!: (response: Response) => void;
  let markRefreshStarted!: () => void;
  const refreshStarted = new Promise<void>((resolve) => {
    markRefreshStarted = resolve;
  });
  const revoked: Array<{ token: string; hint: string }> = [];
  let walletRequests = 0;
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    now: () => 100_000,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.token_endpoint) {
        markRefreshStarted();
        return await new Promise<Response>((resolve) => {
          resolveRefresh = resolve;
        });
      }
      if (url === METADATA.revocation_endpoint) {
        const body = new URLSearchParams(String(init?.body));
        revoked.push({
          token: body.get("token") ?? "",
          hint: body.get("token_type_hint") ?? "",
        });
        return new Response(null, { status: 200 });
      }
      if (url.endsWith("/oauth2/v1/chat/completions")) {
        walletRequests++;
        return sse([
          'data: {"choices":[{"delta":{"content":"must not run"}}]}\n\n',
          "data: [DONE]\n\n",
        ]);
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  const completion = client.streamCompletion({ model: "live-chat", messages: [] });
  await refreshStarted;
  const disconnecting = client.disconnect();
  await Promise.resolve();
  resolveRefresh(
    json({
      access_token: "rotated-access",
      refresh_token: "rotated-refresh",
      expires_in: 600,
    }),
  );

  expect(await disconnecting).toEqual({ revoked: true });
  await expect(completion).rejects.toThrow("disconnected");
  expect(walletRequests).toBe(0);
  expect(store.value).toBeNull();
  expect(revoked).toEqual([
    { token: "rotated-refresh", hint: "refresh_token" },
    { token: "rotated-access", hint: "access_token" },
  ]);
});

test("disconnect fails closed when the secure store does not actually clear", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "access-sensitive",
    refreshToken: "refresh-sensitive",
  } satisfies AiPassTokenBundle);
  store.delete = async () => {
    store.clears++;
    // Simulate a credential backend that reports no exception but retains the item.
    return false;
  };
  let walletRequests = 0;
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    fetchImpl: async (input) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.revocation_endpoint) return new Response(null, { status: 200 });
      if (url.endsWith("/oauth2/v1/chat/completions")) {
        walletRequests++;
        return sse([
          'data: {"choices":[{"delta":{"content":"must not run"}}]}\n\n',
          "data: [DONE]\n\n",
        ]);
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  await expect(client.disconnect()).rejects.toThrow("secure storage");
  await expect(
    client.streamCompletion({ model: "live-chat", messages: [] }),
  ).rejects.toThrow("disconnected");
  expect(walletRequests).toBe(0);
  expect(store.value).not.toContain("sensitive");
});

test("a failed disconnect remains blocked after the AI Pass client is recreated", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "retained-access",
    refreshToken: "retained-refresh",
  } satisfies AiPassTokenBundle);
  store.allowWrite = false;
  store.delete = async () => {
    store.clears++;
    return false;
  };
  const blockStore = {
    value: false,
    get() {
      return this.value;
    },
    set(value: boolean) {
      this.value = value;
      return true;
    },
  };
  const clientOptions = {
    clientId: () => "protected-test-client",
    store,
    blockStore,
  };
  const first = createAiPassClient({
    ...clientOptions,
    fetchImpl: async (input) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.revocation_endpoint) return new Response(null, { status: 503 });
      throw new Error(`unexpected request: ${url}`);
    },
  });

  await expect(first.disconnect()).rejects.toThrow("secure storage");

  let walletRequests = 0;
  const recreated = createAiPassClient({
    ...clientOptions,
    fetchImpl: async (input) => {
      if (String(input).endsWith("/oauth2/v1/chat/completions")) {
        walletRequests++;
        return sse([
          'data: {"choices":[{"delta":{"content":"must not run"}}]}\n\n',
          "data: [DONE]\n\n",
        ]);
      }
      throw new Error(`unexpected request: ${String(input)}`);
    },
  });

  await expect(
    recreated.streamCompletion({ model: "live-chat", messages: [] }),
  ).rejects.toThrow("disconnected");
  expect(walletRequests).toBe(0);
  expect(blockStore.value).toBe(true);
  expect(store.value).toContain("retained-access");
});

test("cancelling a completion aborts the upstream wallet-billed request", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "fresh-access",
    refreshToken: "refresh",
    expiresAt: Date.now() + 60_000,
  } satisfies AiPassTokenBundle);
  let upstreamAborted = false;
  let markStarted!: () => void;
  const started = new Promise<void>((resolve) => {
    markStarted = resolve;
  });
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    fetchImpl: async (input, init) => {
      if (String(input).endsWith("/oauth2/v1/chat/completions")) {
        return await new Promise<Response>((_resolve, reject) => {
          markStarted();
          init?.signal?.addEventListener(
            "abort",
            () => {
              upstreamAborted = true;
              reject(init.signal?.reason ?? new DOMException("aborted", "AbortError"));
            },
            { once: true },
          );
        });
      }
      throw new Error(`unexpected request: ${String(input)}`);
    },
  });
  const controller = new AbortController();
  const pending = client.streamCompletion(
    { model: "live-chat", messages: [] },
    { signal: controller.signal },
  );

  await started;
  controller.abort();

  await expect(pending).rejects.toThrow();
  expect(upstreamAborted).toBe(true);
});

test("disconnect attempts refresh and access revocation, then clears locally even on failure", async () => {
  const store = new MemoryStore();
  store.value = JSON.stringify({
    accessToken: "access-sensitive",
    refreshToken: "refresh-sensitive",
    expiresAt: Date.now() + 60_000,
  } satisfies AiPassTokenBundle);
  const revokedHints: string[] = [];
  const client = createAiPassClient({
    clientId: () => "protected-test-client",
    store,
    fetchImpl: async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/oauth-authorization-server")) return json(METADATA);
      if (url === METADATA.revocation_endpoint) {
        revokedHints.push(new URLSearchParams(String(init?.body)).get("token_type_hint") ?? "");
        return new Response("unavailable", { status: 503 });
      }
      throw new Error(`unexpected request: ${url}`);
    },
  });

  const result = await client.disconnect();

  expect(result).toEqual({ revoked: false });
  expect(revokedHints).toEqual(["refresh_token", "access_token"]);
  expect(store.clears).toBe(1);
  expect(store.value).toBeNull();
});

test("AI Pass credentials and runtime markers never use the plaintext config fallback", () => {
  const previous = process.env.REPOYETI_NO_KEYCHAIN;
  process.env.REPOYETI_NO_KEYCHAIN = "1";
  const cfg: RepoYetiConfig = {
    roots: [],
    port: 7171,
    maxDepth: 6,
    maxRepos: 200,
    ai: {
      providers: {
        aipass: {
          model: "live-model-from-discovery",
          connected: true,
          apiKey: "must-never-be-treated-as-an-aipass-credential",
        },
      },
      defaultProvider: "aipass",
    },
  };
  try {
    saveConfig(cfg);
    const persisted = readFileSync(join(CONFIG_DIR, "config.json"), "utf8");
    expect(persisted).toContain("live-model-from-discovery");
    expect(persisted).not.toContain("must-never");
    expect(persisted).not.toContain("connected");
    expect(persisted).not.toContain("accessToken");
    expect(persisted).not.toContain("refreshToken");
  } finally {
    if (previous === undefined) delete process.env.REPOYETI_NO_KEYCHAIN;
    else process.env.REPOYETI_NO_KEYCHAIN = previous;
  }
});
