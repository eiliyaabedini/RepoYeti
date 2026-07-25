/**
 * AI Pass account transport for the daemon.
 *
 * This is deliberately separate from the BYOK adapters: AI Pass is an OAuth account connection,
 * never an API-key field. The browser only starts the redirect and receives connection status;
 * this module owns PKCE, bearer tokens, refresh rotation, model discovery, streamed generation,
 * cancellation, and revocation outside the webview.
 */
import { createHash, randomBytes } from "node:crypto";
import { AIPASS_TOKEN_BUNDLE, deleteSecret, getSecret, setSecret } from "./secrets.ts";
import type { AiModel } from "./ai/adapters.ts";

const ISSUER = "https://aipass.one";
const DISCOVERY_URL = `${ISSUER}/.well-known/oauth-authorization-server`;
const MODELS_URL = `${ISSUER}/oauth2/v1/models?detailed=true`;
const CHAT_URL = `${ISSUER}/oauth2/v1/chat/completions`;
const CALLBACK_PATH = "/aipass/oauth/callback";
const SCOPES = "api:access profile:read";
const TX_TTL_MS = 10 * 60_000;
const MAX_TRANSACTIONS = 64;
const REQUEST_TIMEOUT_MS = 15_000;
const STREAM_TIMEOUT_MS = 45_000;
const MAX_METADATA_BYTES = 64 * 1024;
const MAX_TOKEN_BYTES = 64 * 1024;
const MAX_USERINFO_BYTES = 64 * 1024;
const MAX_MODELS_BYTES = 1024 * 1024;
const MAX_ERROR_BYTES = 16 * 1024;
const MAX_REQUEST_BYTES = 256 * 1024;
const MAX_STREAM_BYTES = 1024 * 1024;
const MAX_COMPLETION_CHARS = 128 * 1024;
const MAX_CLIENT_ID_CHARS = 512;
const MAX_TOKEN_CHARS = 64 * 1024;
const MAX_MODEL_ID_CHARS = 512;
const MAX_MODEL_LABEL_CHARS = 512;
const REFRESH_EARLY_MS = 30_000;

export type AiPassCode =
  | "NOT_CONFIGURED"
  | "AI_AUTH_FAILED"
  | "AI_BAD_REQUEST"
  | "AI_RATE_LIMITED"
  | "AI_UNREACHABLE"
  | "AI_ERROR";

export class AiPassError extends Error {
  constructor(
    public readonly code: AiPassCode,
    message: string,
    public readonly status = 400,
  ) {
    super(message);
    this.name = "AiPassError";
  }
}

export interface AiPassTokenBundle {
  accessToken: string;
  refreshToken?: string;
  expiresAt?: number;
  tokenType?: string;
}

/** Narrow seam around the native credential store. `set` must report whether persistence worked:
 *  OAuth completion and refresh both fail closed instead of falling back to config.json. */
export interface AiPassSecretStore {
  get(): Promise<string | null>;
  set(value: string): Promise<boolean>;
  delete(): Promise<void>;
}

type FetchLike = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

interface Metadata {
  issuer: string;
  authorization_endpoint: string;
  token_endpoint: string;
  userinfo_endpoint: string;
  revocation_endpoint: string;
  code_challenge_methods_supported: string[];
  token_endpoint_auth_methods_supported: string[];
}

interface Transaction {
  verifier: string;
  redirectUri: string;
  expiresAt: number;
}

export interface AiPassClientOptions {
  /** Protected runtime/build configuration getter. Never returned by status APIs or logged. */
  clientId?: () => string | undefined;
  fetchImpl?: FetchLike;
  store?: AiPassSecretStore;
  now?: () => number;
}

const nativeStore: AiPassSecretStore = {
  get: () => getSecret(AIPASS_TOKEN_BUNDLE),
  set: (value) => setSecret(AIPASS_TOKEN_BUNDLE, value),
  delete: () => deleteSecret(AIPASS_TOKEN_BUNDLE),
};

function configuredClientId(): string | undefined {
  return process.env.REPOYETI_AIPASS_CLIENT_ID?.trim() || undefined;
}

function base64url(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString("base64url");
}

function callbackUri(origin: string): string {
  let parsed: URL;
  try {
    parsed = new URL(origin);
  } catch {
    throw new AiPassError("AI_BAD_REQUEST", "invalid AI Pass callback origin");
  }
  if (
    parsed.username ||
    parsed.password ||
    parsed.pathname !== "/" ||
    parsed.search ||
    parsed.hash ||
    (parsed.protocol !== "https:" &&
      !(parsed.protocol === "http:" &&
        (parsed.hostname === "127.0.0.1" ||
          parsed.hostname === "localhost" ||
          parsed.hostname === "[::1]")))
  ) {
    throw new AiPassError(
      "AI_BAD_REQUEST",
      "AI Pass requires HTTPS, except for a loopback callback",
    );
  }
  return `${parsed.origin}${CALLBACK_PATH}`;
}

function tokenBundle(raw: string | null): AiPassTokenBundle | null {
  if (!raw || Buffer.byteLength(raw) > MAX_TOKEN_BYTES) return null;
  try {
    const value = JSON.parse(raw) as Record<string, unknown>;
    if (
      typeof value.accessToken !== "string" ||
      !value.accessToken ||
      value.accessToken.length > MAX_TOKEN_CHARS
    ) {
      return null;
    }
    if (
      value.refreshToken != null &&
      (typeof value.refreshToken !== "string" ||
        value.refreshToken.length > MAX_TOKEN_CHARS)
    ) {
      return null;
    }
    if (value.expiresAt != null && typeof value.expiresAt !== "number") return null;
    return {
      accessToken: value.accessToken,
      ...(typeof value.refreshToken === "string" && value.refreshToken
        ? { refreshToken: value.refreshToken }
        : {}),
      ...(typeof value.expiresAt === "number" ? { expiresAt: value.expiresAt } : {}),
      ...(typeof value.tokenType === "string" ? { tokenType: value.tokenType } : {}),
    };
  } catch {
    return null;
  }
}

function serializeTokens(tokens: AiPassTokenBundle): string {
  return JSON.stringify(tokens);
}

function abortScope(external: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();
  const onAbort = (): void => controller.abort(external?.reason);
  if (external?.aborted) onAbort();
  else external?.addEventListener("abort", onAbort, { once: true });
  const timer = setTimeout(
    () => controller.abort(new DOMException("AI Pass request timed out", "TimeoutError")),
    timeoutMs,
  );
  return {
    signal: controller.signal,
    cleanup: (): void => {
      clearTimeout(timer);
      external?.removeEventListener("abort", onAbort);
    },
  };
}

async function boundedText(response: Response, maxBytes: number): Promise<string> {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maxBytes) {
    await response.body?.cancel();
    throw new AiPassError("AI_ERROR", "AI Pass returned an oversized response", 502);
  }
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes) {
        await reader.cancel();
        throw new AiPassError("AI_ERROR", "AI Pass returned an oversized response", 502);
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const out = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(out);
}

async function boundedJson(response: Response, maxBytes: number): Promise<unknown> {
  const text = await boundedText(response, maxBytes);
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    throw new AiPassError("AI_ERROR", "AI Pass returned an invalid response", 502);
  }
}

function statusError(response: Response, context: string): AiPassError {
  if (response.status === 401 || response.status === 403) {
    return new AiPassError(
      "AI_AUTH_FAILED",
      "AI Pass authorization expired; connect the account again",
      response.status,
    );
  }
  if (response.status === 429) {
    return new AiPassError(
      "AI_RATE_LIMITED",
      "AI Pass is rate limiting this request; retry shortly",
      429,
    );
  }
  if (response.status >= 400 && response.status < 500) {
    return new AiPassError("AI_BAD_REQUEST", `AI Pass rejected the ${context}`, response.status);
  }
  return new AiPassError("AI_ERROR", `AI Pass could not complete the ${context}`, 502);
}

function parseMetadata(raw: unknown): Metadata {
  const value = (raw ?? {}) as Record<string, unknown>;
  const metadata: Metadata = {
    issuer: String(value.issuer ?? ""),
    authorization_endpoint: String(value.authorization_endpoint ?? ""),
    token_endpoint: String(value.token_endpoint ?? ""),
    userinfo_endpoint: String(value.userinfo_endpoint ?? ""),
    revocation_endpoint: String(value.revocation_endpoint ?? ""),
    code_challenge_methods_supported: Array.isArray(value.code_challenge_methods_supported)
      ? value.code_challenge_methods_supported.map(String)
      : [],
    token_endpoint_auth_methods_supported: Array.isArray(
      value.token_endpoint_auth_methods_supported,
    )
      ? value.token_endpoint_auth_methods_supported.map(String)
      : [],
  };
  const endpoints = [
    metadata.authorization_endpoint,
    metadata.token_endpoint,
    metadata.userinfo_endpoint,
    metadata.revocation_endpoint,
  ];
  if (
    metadata.issuer !== ISSUER ||
    !endpoints.every((endpoint) => {
      try {
        return new URL(endpoint).origin === ISSUER;
      } catch {
        return false;
      }
    }) ||
    !metadata.code_challenge_methods_supported.includes("S256") ||
    !metadata.token_endpoint_auth_methods_supported.includes("none")
  ) {
    throw new AiPassError("AI_ERROR", "AI Pass OAuth metadata failed validation", 502);
  }
  return metadata;
}

/** Normalize both documented OpenAI `{ object:"list", data:[...] }` payloads and the legacy
 * string-array response. Detailed entries are restricted to the chat-completions method. */
export function parseAiPassModels(raw: unknown): AiModel[] {
  const source = Array.isArray(raw)
    ? raw
    : Array.isArray((raw as { data?: unknown } | null)?.data)
      ? ((raw as { data: unknown[] }).data ?? [])
      : [];
  const seen = new Set<string>();
  const models: AiModel[] = [];
  for (const entry of source) {
    const record =
      typeof entry === "string" ? null : ((entry ?? {}) as Record<string, unknown>);
    const id =
      typeof entry === "string"
        ? entry.trim()
        : typeof record?.id === "string"
          ? record.id.trim()
          : "";
    if (!id || id.length > MAX_MODEL_ID_CHARS || seen.has(id)) continue;
    const methods = record?.methods;
    if (
      Array.isArray(methods) &&
      methods.length > 0 &&
      !methods.map(String).includes("chat_completions")
    ) {
      continue;
    }
    seen.add(id);
    const name =
      typeof record?.name === "string"
        ? record.name.trim().slice(0, MAX_MODEL_LABEL_CHARS)
        : "";
    models.push({ id, label: name || id });
  }
  models.sort((a, b) => a.id.localeCompare(b.id));
  return models;
}

function completionDelta(raw: unknown): string {
  const choice = (raw as { choices?: Array<Record<string, unknown>> } | null)?.choices?.[0];
  const delta = choice?.delta as { content?: unknown } | undefined;
  return typeof delta?.content === "string" ? delta.content : "";
}

export function createAiPassClient(options: AiPassClientOptions = {}) {
  const clientIdFor = options.clientId ?? configuredClientId;
  const fetchImpl = options.fetchImpl ?? fetch;
  const store = options.store ?? nativeStore;
  const now = options.now ?? Date.now;
  const transactions = new Map<string, Transaction>();
  let metadataCache: Metadata | null = null;
  let refreshPromise: Promise<AiPassTokenBundle> | null = null;

  const clientId = (): string => {
    const value = clientIdFor()?.trim();
    if (!value || value.length > MAX_CLIENT_ID_CHARS) {
      throw new AiPassError(
        "NOT_CONFIGURED",
        "AI Pass is unavailable because this build has no configured public client",
      );
    }
    return value;
  };

  const request = async (
    url: string,
    init: RequestInit,
    maxBytes: number,
    timeoutMs = REQUEST_TIMEOUT_MS,
  ): Promise<{ response: Response; json: unknown }> => {
    const scope = abortScope(init.signal ?? undefined, timeoutMs);
    try {
      const response = await fetchImpl(url, { ...init, signal: scope.signal });
      const body = await boundedJson(response, maxBytes);
      return { response, json: body };
    } catch (error) {
      if (error instanceof AiPassError) throw error;
      if (scope.signal.aborted && init.signal?.aborted) throw error;
      throw new AiPassError("AI_UNREACHABLE", "could not reach AI Pass", 502);
    } finally {
      scope.cleanup();
    }
  };

  const metadata = async (): Promise<Metadata> => {
    if (metadataCache) return metadataCache;
    const { response, json } = await request(
      DISCOVERY_URL,
      { method: "GET", headers: { accept: "application/json" } },
      MAX_METADATA_BYTES,
    );
    if (!response.ok) throw statusError(response, "OAuth discovery");
    metadataCache = parseMetadata(json);
    return metadataCache;
  };

  const loadTokens = async (): Promise<AiPassTokenBundle | null> => tokenBundle(await store.get());

  const persistTokens = async (tokens: AiPassTokenBundle): Promise<void> => {
    if (!(await store.set(serializeTokens(tokens)))) {
      throw new AiPassError(
        "NOT_CONFIGURED",
        "AI Pass requires native secure storage on this system",
      );
    }
  };

  const revokeOne = async (
    endpoint: string,
    id: string,
    token: string,
    hint: "access_token" | "refresh_token",
  ): Promise<boolean> => {
    const body = new URLSearchParams({ token, token_type_hint: hint, client_id: id });
    try {
      const scope = abortScope(undefined, REQUEST_TIMEOUT_MS);
      try {
        const response = await fetchImpl(endpoint, {
          method: "POST",
          headers: {
            "content-type": "application/x-www-form-urlencoded",
            accept: "application/json",
          },
          body,
          signal: scope.signal,
        });
        await boundedText(response, MAX_ERROR_BYTES);
        return response.ok;
      } finally {
        scope.cleanup();
      }
    } catch {
      return false;
    }
  };

  const refresh = async (): Promise<AiPassTokenBundle> => {
    if (refreshPromise) return refreshPromise;
    refreshPromise = (async () => {
      const previous = await loadTokens();
      if (!previous?.refreshToken) {
        throw new AiPassError(
          "AI_AUTH_FAILED",
          "AI Pass authorization expired; connect the account again",
          401,
        );
      }
      const doc = await metadata();
      const { response, json } = await request(
        doc.token_endpoint,
        {
          method: "POST",
          headers: { "content-type": "application/json", accept: "application/json" },
          body: JSON.stringify({
            grantType: "refresh_token",
            refreshToken: previous.refreshToken,
            clientId: clientId(),
          }),
        },
        MAX_TOKEN_BYTES,
      );
      if (!response.ok) {
        if (response.status === 400 || response.status === 401) await store.delete();
        throw statusError(response, "token refresh");
      }
      const value = (json ?? {}) as Record<string, unknown>;
      const accessToken =
        typeof value.access_token === "string" ? value.access_token : "";
      if (!accessToken || accessToken.length > MAX_TOKEN_CHARS) {
        throw new AiPassError("AI_ERROR", "AI Pass returned no access token", 502);
      }
      const returnedRefresh =
        typeof value.refresh_token === "string" ? value.refresh_token : "";
      if (returnedRefresh.length > MAX_TOKEN_CHARS) {
        throw new AiPassError("AI_ERROR", "AI Pass returned an invalid refresh token", 502);
      }
      const expiresIn = Number(value.expires_in);
      const rotated: AiPassTokenBundle = {
        accessToken,
        refreshToken: returnedRefresh || previous.refreshToken,
        ...(Number.isFinite(expiresIn) && expiresIn > 0
          ? { expiresAt: now() + expiresIn * 1000 }
          : {}),
        ...(typeof value.token_type === "string" ? { tokenType: value.token_type } : {}),
      };
      // One native-store replacement makes access + rotated refresh token indivisible.
      try {
        await persistTokens(rotated);
      } catch (error) {
        // The provider may already have invalidated the previous refresh token. Keeping that
        // stale bundle would falsely present the account as connected after rotation failed.
        await store.delete();
        throw error;
      }
      return rotated;
    })();
    try {
      return await refreshPromise;
    } finally {
      refreshPromise = null;
    }
  };

  const accessToken = async (): Promise<string> => {
    const current = await loadTokens();
    if (!current) {
      throw new AiPassError("NOT_CONFIGURED", "Connect AI Pass before using its wallet", 404);
    }
    if (current.expiresAt != null && current.expiresAt <= now() + REFRESH_EARLY_MS) {
      return (await refresh()).accessToken;
    }
    return current.accessToken;
  };

  const fetchModels = async (token: string, retry = true): Promise<AiModel[]> => {
    const { response, json } = await request(
      MODELS_URL,
      { method: "GET", headers: { authorization: `Bearer ${token}`, accept: "application/json" } },
      MAX_MODELS_BYTES,
    );
    if (response.status === 401 && retry) return fetchModels((await refresh()).accessToken, false);
    if (!response.ok) throw statusError(response, "model discovery");
    return parseAiPassModels(json);
  };

  const beginAuthorization = async (origin: string): Promise<string> => {
    const id = clientId();
    const doc = await metadata();
    const verifier = base64url(randomBytes(32));
    const challenge = base64url(createHash("sha256").update(verifier).digest());
    const state = base64url(randomBytes(32));
    const redirectUri = callbackUri(origin);
    for (const [key, tx] of transactions) {
      if (tx.expiresAt <= now()) transactions.delete(key);
    }
    while (transactions.size >= MAX_TRANSACTIONS) {
      const oldest = transactions.keys().next().value as string | undefined;
      if (!oldest) break;
      transactions.delete(oldest);
    }
    transactions.set(state, { verifier, redirectUri, expiresAt: now() + TX_TTL_MS });
    const url = new URL(doc.authorization_endpoint);
    url.searchParams.set("response_type", "code");
    url.searchParams.set("client_id", id);
    url.searchParams.set("redirect_uri", redirectUri);
    url.searchParams.set("scope", SCOPES);
    url.searchParams.set("state", state);
    url.searchParams.set("code_challenge", challenge);
    url.searchParams.set("code_challenge_method", "S256");
    return url.toString();
  };

  const cancelAuthorization = (state: string): void => {
    if (state && state.length <= 256) transactions.delete(state);
  };

  const completeAuthorization = async (
    code: string,
    state: string,
  ): Promise<{ models: AiModel[] }> => {
    if (state.length > 256 || code.length > 4096) {
      throw new AiPassError("AI_BAD_REQUEST", "Invalid AI Pass callback parameters");
    }
    const tx = transactions.get(state);
    transactions.delete(state);
    if (!tx || tx.expiresAt <= now()) {
      throw new AiPassError("AI_BAD_REQUEST", "This AI Pass connection expired; start again");
    }
    if (!code) throw new AiPassError("AI_BAD_REQUEST", "AI Pass returned no authorization code");
    const doc = await metadata();
    const { response, json } = await request(
      doc.token_endpoint,
      {
        method: "POST",
        headers: { "content-type": "application/json", accept: "application/json" },
        body: JSON.stringify({
          grantType: "authorization_code",
          code,
          codeVerifier: tx.verifier,
          clientId: clientId(),
          redirectUri: tx.redirectUri,
        }),
      },
      MAX_TOKEN_BYTES,
    );
    if (!response.ok) throw statusError(response, "authorization-code exchange");
    const value = (json ?? {}) as Record<string, unknown>;
    const access = typeof value.access_token === "string" ? value.access_token : "";
    const refreshToken =
      typeof value.refresh_token === "string" ? value.refresh_token : "";
    if (!access || access.length > MAX_TOKEN_CHARS) {
      throw new AiPassError("AI_ERROR", "AI Pass returned no access token", 502);
    }
    if (refreshToken.length > MAX_TOKEN_CHARS) {
      throw new AiPassError("AI_ERROR", "AI Pass returned an invalid refresh token", 502);
    }
    const expiresIn = Number(value.expires_in);
    const tokens: AiPassTokenBundle = {
      accessToken: access,
      ...(refreshToken ? { refreshToken } : {}),
      ...(Number.isFinite(expiresIn) && expiresIn > 0
        ? { expiresAt: now() + expiresIn * 1000 }
        : {}),
      ...(typeof value.token_type === "string" ? { tokenType: value.token_type } : {}),
    };

    const userinfo = await request(
      doc.userinfo_endpoint,
      {
        method: "GET",
        headers: { authorization: `Bearer ${access}`, accept: "application/json" },
      },
      MAX_USERINFO_BYTES,
    );
    if (!userinfo.response.ok) throw statusError(userinfo.response, "account verification");
    // This grant is intentionally not stored until verification succeeds, so a 401 here must
    // fail the new connection rather than trying to refresh an unrelated prior bundle.
    const models = await fetchModels(access, false);
    try {
      await persistTokens(tokens);
    } catch (error) {
      // The grant exists but cannot be stored safely. Best-effort revoke both halves, retain none.
      if (refreshToken) {
        await revokeOne(doc.revocation_endpoint, clientId(), refreshToken, "refresh_token");
      }
      await revokeOne(doc.revocation_endpoint, clientId(), access, "access_token");
      await store.delete();
      throw error;
    }
    return { models };
  };

  const streamCompletion = async (
    body: Record<string, unknown>,
    streamOptions: { signal?: AbortSignal; timeoutMs?: number } = {},
  ): Promise<string> => {
    const requestBody = JSON.stringify({ ...body, stream: true });
    if (Buffer.byteLength(requestBody) > MAX_REQUEST_BYTES) {
      throw new AiPassError("AI_BAD_REQUEST", "AI Pass request is too large");
    }
    const scope = abortScope(
      streamOptions.signal,
      streamOptions.timeoutMs ?? STREAM_TIMEOUT_MS,
    );
    let response: Response | null = null;
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    try {
      const send = (token: string): Promise<Response> =>
        fetchImpl(CHAT_URL, {
          method: "POST",
          headers: {
            authorization: `Bearer ${token}`,
            "content-type": "application/json",
            accept: "text/event-stream",
          },
          body: requestBody,
          signal: scope.signal,
        });
      response = await send(await accessToken());
      if (response.status === 401) {
        await response.body?.cancel();
        response = await send((await refresh()).accessToken);
      }
      if (!response.ok) {
        await boundedText(response, MAX_ERROR_BYTES);
        throw statusError(response, "wallet request");
      }
      if (
        !response.headers
          .get("content-type")
          ?.toLowerCase()
          .startsWith("text/event-stream")
      ) {
        await boundedText(response, MAX_ERROR_BYTES);
        throw new AiPassError("AI_ERROR", "AI Pass returned an invalid completion stream", 502);
      }
      if (!response.body) {
        throw new AiPassError("AI_ERROR", "AI Pass returned no completion stream", 502);
      }
      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let output = "";
      let bytes = 0;
      let finished = false;
      while (!finished) {
        const chunk = await reader.read();
        if (chunk.done) break;
        bytes += chunk.value.byteLength;
        if (bytes > MAX_STREAM_BYTES) {
          throw new AiPassError("AI_ERROR", "AI Pass completion stream was too large", 502);
        }
        buffer += decoder.decode(chunk.value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const data = trimmed.slice(5).trimStart();
          if (data === "[DONE]") {
            finished = true;
            break;
          }
          let parsed: unknown;
          try {
            parsed = JSON.parse(data);
          } catch {
            continue;
          }
          if ((parsed as { error?: unknown } | null)?.error) {
            throw new AiPassError("AI_ERROR", "AI Pass stopped the completion", 502);
          }
          output += completionDelta(parsed);
          if (output.length > MAX_COMPLETION_CHARS) {
            throw new AiPassError("AI_ERROR", "AI Pass completion was too large", 502);
          }
        }
      }
      if (!output) throw new AiPassError("AI_ERROR", "AI Pass returned an empty completion", 502);
      return output;
    } catch (error) {
      if (error instanceof AiPassError) throw error;
      if (scope.signal.aborted && streamOptions.signal?.aborted) throw error;
      if (scope.signal.aborted) {
        throw new AiPassError("AI_UNREACHABLE", "AI Pass completion timed out", 504);
      }
      throw new AiPassError("AI_UNREACHABLE", "could not reach AI Pass", 502);
    } finally {
      if (reader) {
        try {
          await reader.cancel();
        } catch {
          /* the upstream may already be closed */
        }
        reader.releaseLock();
      } else if (response?.body) {
        try {
          await response.body.cancel();
        } catch {
          /* best-effort upstream cancellation */
        }
      }
      scope.cleanup();
    }
  };

  const disconnect = async (): Promise<{ revoked: boolean }> => {
    const tokens = await loadTokens();
    let revoked = true;
    try {
      if (tokens) {
        const doc = await metadata();
        const id = clientId();
        if (tokens.refreshToken) {
          revoked =
            (await revokeOne(
              doc.revocation_endpoint,
              id,
              tokens.refreshToken,
              "refresh_token",
            )) && revoked;
        }
        revoked =
          (await revokeOne(
            doc.revocation_endpoint,
            id,
            tokens.accessToken,
            "access_token",
          )) && revoked;
      }
    } catch {
      revoked = false;
    } finally {
      await store.delete();
    }
    return { revoked };
  };

  return {
    beginAuthorization,
    cancelAuthorization,
    completeAuthorization,
    isConnected: async (): Promise<boolean> => (await loadTokens()) !== null,
    listModels: async (): Promise<AiModel[]> => fetchModels(await accessToken()),
    streamCompletion,
    disconnect,
  };
}

export type AiPassClient = ReturnType<typeof createAiPassClient>;

/** Production singleton: server-side only, backed by Bun.secrets. */
export const aiPassClient = createAiPassClient();
