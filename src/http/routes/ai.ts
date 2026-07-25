import type { Hono, Context } from "hono";
import type { ContentfulStatusCode } from "hono/utils/http-status";
import type { Deps } from "../deps.ts";
import {
  redactAi,
  saveConfig,
  AI_PROVIDERS,
  AI_CATALOG,
  resolveApiKey,
  resolveModel,
  providerConfigured,
  effectiveDefaultProvider,
  DEFAULT_DIFF_DETAIL,
  type RepoYetiConfig,
  type AiProviderId,
} from "../../config.ts";
import {
  listModels,
  generateCommitMessage,
  generateCommitPlan,
  heuristicPlan,
  clearRateGate,
  AiError,
} from "../../ai.ts";
import { jsonError, type ApiErrorCode } from "../../contract.ts";
import { setSecret, deleteSecret, aiKeyName } from "../../secrets.ts";
import {
  parseBody,
  AiSettingsSchema,
  ProviderUpdateSchema,
  ConnectSchema,
  CommitMessageSchema,
  CommitPlanSchema,
} from "../../schemas.ts";
import { collectRepoDiff, collectRepoPathsDiff, planCommitInput } from "../../service/index.ts";
import { requireId } from "../respond.ts";
import { effectiveGuest } from "../../auth.ts";
import { AiPassError } from "../../aipass.ts";

const GUEST_AI_WINDOW_MS = 60_000;
const GUEST_AI_MAX_PER_WINDOW = 10;
const GUEST_AI_MAX_CONCURRENT = 2;

interface GuestAiUsage {
  windowStartedAt: number;
  used: number;
  active: number;
}

function publicOrigin(c: Context): string {
  const url = new URL(c.req.url);
  const forwarded = c.req.header("x-forwarded-proto")?.split(",")[0]?.trim();
  if (forwarded) url.protocol = `${forwarded}:`;
  return url.origin;
}

function escapeHtml(value: string): string {
  return value.replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ]!,
  );
}

function connectionPage(ok: boolean, message: string): string {
  const title = ok ? "AI Pass connected" : "Could not connect AI Pass";
  const icon = ok ? "✓" : "×";
  const color = ok ? "#3ddc84" : "#ff6b6b";
  return `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta http-equiv="refresh" content="2;url=/?aipass=${ok ? "connected" : "error"}">
<title>RepoYeti — ${title}</title></head>
<body style="margin:0;background:#0e0e12;color:#e6e6ea;font-family:system-ui,sans-serif;display:grid;place-items:center;min-height:100vh">
<main style="max-width:360px;text-align:center;padding:24px">
<div style="font-size:42px;color:${color}">${icon}</div>
<h2 style="margin:12px 0 8px">${title}</h2>
<p style="color:#9a9aa6;font-size:14px;line-height:1.5">${escapeHtml(message)}</p>
<a href="/" style="display:inline-block;margin-top:14px;color:#3ddc84">Return to RepoYeti</a>
</main></body></html>`;
}

function connectionResponse(
  c: Context,
  ok: boolean,
  message: string,
  status: ContentfulStatusCode = 200,
): Response {
  c.header("cache-control", "no-store");
  c.header("referrer-policy", "no-referrer");
  c.header(
    "content-security-policy",
    "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
  );
  return c.html(connectionPage(ok, message), status);
}

export function register(app: Hono, { cfg, aiPass }: Deps): void {
  // ── AI: bring-your-own-key commit messages ──────────────────────────────────
  // The daemon makes every provider call; the owner's key never reaches the browser.
  // `cfg` is mutated in place AND persisted so a running daemon picks up new keys.
  const parseProvider = (c: Context): AiProviderId | null => {
    const p = c.req.param("provider") ?? "";
    return (AI_PROVIDERS as readonly string[]).includes(p) ? (p as AiProviderId) : null;
  };
  const ensureAi = (): NonNullable<RepoYetiConfig["ai"]> => (cfg.ai ??= { providers: {} });
  const providerLabel = (id: AiProviderId): string => AI_CATALOG.find((e) => e.id === id)?.label ?? id;
  const guestAiUsage = new Map<string, GuestAiUsage>();
  const enterGuestAi = (c: Context): Response | (() => void) | null => {
    const guest = effectiveGuest(c, cfg);
    if (!guest) return null;
    if (cfg.ai?.commitEnabled === false) {
      return jsonError(c, "FORBIDDEN", "AI commit generation is disabled by the owner", 403);
    }
    const now = Date.now();
    let usage = guestAiUsage.get(guest.id);
    if (!usage || now - usage.windowStartedAt >= GUEST_AI_WINDOW_MS) {
      usage = { windowStartedAt: now, used: 0, active: 0 };
      guestAiUsage.set(guest.id, usage);
    }
    if (usage.active >= GUEST_AI_MAX_CONCURRENT) {
      return jsonError(c, "AI_RATE_LIMITED", "too many AI requests are already running for this share link");
    }
    if (usage.used >= GUEST_AI_MAX_PER_WINDOW) {
      return jsonError(c, "AI_RATE_LIMITED", "this share link has reached its AI request limit; retry shortly");
    }
    usage.used++;
    usage.active++;
    return () => {
      usage!.active = Math.max(0, usage!.active - 1);
    };
  };
  // Turn a raw AiError into a client message. A 401/403 (AI_AUTH_FAILED) is enriched with WHICH
  // provider's key failed, so the owner isn't left staring at a bare "invalid or unauthorized key"
  // wondering what to fix.
  const aiErr = (c: Context, e: unknown, provider?: AiProviderId) => {
    if (e instanceof AiPassError) {
      return jsonError(c, e.code as ApiErrorCode, e.message);
    }
    if (e instanceof AiError) {
      if (e.code === "AI_AUTH_FAILED" && provider) {
        const label = providerLabel(provider);
        const message =
          provider === "aipass"
            ? "AI Pass authorization expired. Connect AI Pass again in Settings → AI."
            : `${label} rejected the API key. Update your ${label} key in Settings → AI.`;
        return jsonError(c, e.code as ApiErrorCode, message);
      }
      return jsonError(c, e.code as ApiErrorCode, e.message);
    }
    return jsonError(c, "AI_ERROR", e instanceof Error ? e.message : String(e));
  };

  // Static provider catalog — safe display metadata (no secrets).
  // Separate endpoint so the UI can cache it independently of per-user settings.
  app.get("/api/ai/catalog", (c) => c.json({ catalog: AI_CATALOG }));

  // OAuth account connection. The public client id comes only from protected runtime/build
  // configuration and appears solely in the OAuth redirect (as the protocol requires), never in
  // the PWA bundle, ordinary JSON APIs, logs, or persisted app config.
  app.get("/api/ai/aipass/connect", async (c) => {
    try {
      return c.redirect(await aiPass.beginAuthorization(publicOrigin(c)));
    } catch (error) {
      const message =
        error instanceof AiPassError
          ? error.message
          : "AI Pass connection could not be started.";
      return connectionResponse(c, false, message, 400);
    }
  });

  // Public callback: possession of the one-use, 256-bit state is the authorization. The token
  // exchange, userinfo validation, live model discovery, and secure-store write all stay inside
  // the daemon. No bearer token is placed in this response, its URL, or a browser cookie.
  app.get("/aipass/oauth/callback", async (c) => {
    const code = c.req.query("code") ?? "";
    const state = c.req.query("state") ?? "";
    if (c.req.query("error")) {
      aiPass.cancelAuthorization(state);
      return connectionResponse(
        c,
        false,
        "AI Pass did not authorize the connection.",
        400,
      );
    }
    try {
      const { models } = await aiPass.completeAuthorization(code, state);
      const ai = ensureAi();
      const previousEntry = ai.providers.aipass;
      const previousDefault = ai.defaultProvider;
      const previous = ai.providers.aipass?.model ?? null;
      const model =
        previous && models.some((entry) => entry.id === previous)
          ? previous
          : (models[0]?.id ?? null);
      ai.providers.aipass = { connected: true, model };
      if (!ai.defaultProvider && model) ai.defaultProvider = "aipass";
      try {
        saveConfig(cfg);
      } catch {
        if (previousEntry) {
          ai.providers.aipass = { model: previousEntry.model };
        } else {
          delete ai.providers.aipass;
        }
        ai.defaultProvider =
          previousDefault === "aipass"
            ? AI_PROVIDERS.find(
                (provider) =>
                  provider !== "aipass" &&
                  providerConfigured(cfg, provider) &&
                  !!resolveModel(cfg, provider),
              )
            : previousDefault;
        await aiPass.disconnect();
        throw new AiPassError(
          "NOT_CONFIGURED",
          "AI Pass was not connected because RepoYeti could not save its settings",
        );
      }
      return connectionResponse(
        c,
        true,
        "RepoYeti can now use models through your shared AI Pass wallet.",
      );
    } catch (error) {
      const message =
        error instanceof AiPassError
          ? error.message
          : "AI Pass connection could not be completed.";
      return connectionResponse(c, false, message, 400);
    }
  });

  // Minimal capability projection for share-link guests. It answers only whether the owner's
  // daemon can generate and whether the feature is enabled; provider/model/key identity remains
  // owner-only. The actual provider call below already happens here, never in the guest browser.
  app.get("/api/ai/availability", (c) => {
    const provider = effectiveDefaultProvider(cfg);
    return c.json({
      usable: !!(provider && providerConfigured(cfg, provider) && resolveModel(cfg, provider)),
      commitEnabled: cfg.ai?.commitEnabled !== false,
    });
  });

  // Redacted settings — NEVER includes any apiKey.
  app.get("/api/ai/settings", (c) => c.json(redactAi(cfg)));

  // Update commit style and/or the default provider.
  app.put("/api/ai/settings", async (c) => {
    const p = await parseBody(c, AiSettingsSchema);
    if (!p.ok) return p.res;
    const ai = ensureAi();
    if (p.data.style != null) ai.style = p.data.style;
    if (p.data.diffDetail != null) ai.diffDetail = p.data.diffDetail;
    if (typeof p.data.yolo === "boolean") ai.yolo = p.data.yolo;
    if (typeof p.data.commitEnabled === "boolean") ai.commitEnabled = p.data.commitEnabled;
    if (p.data.defaultProvider !== undefined) {
      const dp = p.data.defaultProvider == null ? undefined : (p.data.defaultProvider as AiProviderId);
      if (dp !== undefined && !providerConfigured(cfg, dp)) {
        return jsonError(c, "NOT_CONFIGURED", `${dp} is not connected`);
      }
      ai.defaultProvider = dp;
    }
    saveConfig(cfg);
    return c.json(redactAi(cfg));
  });

  // Connect a provider: validate the key by listing models, then SAVE it.
  app.post("/api/ai/providers/:provider/connect", async (c) => {
    const provider = parseProvider(c);
    if (!provider) return jsonError(c, "BAD_PROVIDER", "unknown provider");
    if (provider === "aipass") {
      return jsonError(
        c,
        "NOT_CONFIGURED",
        "Use Connect AI Pass; AI Pass is an account connection, not an API-key provider",
      );
    }
    const p = await parseBody(c, ConnectSchema);
    if (!p.ok) return p.res;
    const apiKey = (p.data.apiKey ?? "").trim();
    if (!apiKey) return jsonError(c, "NO_KEY", "API key required");
    try {
      const models = await listModels(provider, apiKey);
      const ai = ensureAi();
      const prev = ai.providers[provider]?.model ?? null;
      // Auto-pick a model so it works immediately: keep a still-valid prior choice, else the
      // provider's curated `recommended` model (config.ts AI_CATALOG) when the live list has it,
      // else the first CHAT model (non-chat models are already filtered out in adapters.ts, so
      // models[0] is a safe fallback — no more Groq → Whisper default).
      const recommended = AI_CATALOG.find((e) => e.id === provider)?.recommended;
      const model =
        prev && models.some((m) => m.id === prev)
          ? prev
          : recommended && models.some((m) => m.id === recommended)
            ? recommended
            : (models[0]?.id ?? null);
      // The key bytes go to the OS keychain; config.json (written by saveConfig) keeps only
      // the model. apiKey stays in the in-memory cfg so this running daemon can use it.
      await setSecret(aiKeyName(provider), apiKey);
      ai.providers[provider] = { apiKey, model };
      if (!ai.defaultProvider) ai.defaultProvider = provider;
      // A new key is exactly how an owner fixes a spent quota (upgraded tier / different account),
      // so drop any rate-limit pause we're holding for this provider — otherwise the fix would
      // look like it didn't work until the pause aged out.
      clearRateGate(provider);
      saveConfig(cfg);
      return c.json({ ok: true, models, settings: redactAi(cfg) });
    } catch (e) {
      return aiErr(c, e, provider);
    }
  });

  // Re-list models for an already-connected provider (refresh the dropdown).
  app.get("/api/ai/providers/:provider/models", async (c) => {
    const provider = parseProvider(c);
    if (!provider) return jsonError(c, "BAD_PROVIDER", "unknown provider");
    if (provider === "aipass") {
      if (!providerConfigured(cfg, provider) || !(await aiPass.isConnected())) {
        return jsonError(c, "NOT_CONFIGURED", "Connect AI Pass first", 404);
      }
      try {
        return c.json({ ok: true, models: await aiPass.listModels() });
      } catch (error) {
        return aiErr(c, error, provider);
      }
    }
    const apiKey = resolveApiKey(cfg, provider);
    // 404 (not the default 400): the named provider has no stored key to list models for.
    if (!apiKey) return jsonError(c, "NOT_CONFIGURED", "no key for this provider", 404);
    try {
      return c.json({ ok: true, models: await listModels(provider, apiKey) });
    } catch (e) {
      return aiErr(c, e, provider);
    }
  });

  // Set the selected model and/or mark this provider the default.
  app.put("/api/ai/providers/:provider", async (c) => {
    const provider = parseProvider(c);
    if (!provider) return jsonError(c, "BAD_PROVIDER", "unknown provider");
    if (!providerConfigured(cfg, provider)) {
      return jsonError(c, "NOT_CONFIGURED", "connect this provider first", 404);
    }
    const p = await parseBody(c, ProviderUpdateSchema);
    if (!p.ok) return p.res;
    const ai = ensureAi();
    const entry = ai.providers[provider];
    if (p.data.model !== undefined && entry) entry.model = p.data.model ?? null;
    if (p.data.makeDefault) ai.defaultProvider = provider;
    saveConfig(cfg);
    return c.json(redactAi(cfg));
  });

  // Remove a provider's key (and re-home the default if it pointed here).
  app.delete("/api/ai/providers/:provider", async (c) => {
    const provider = parseProvider(c);
    if (!provider) return jsonError(c, "BAD_PROVIDER", "unknown provider");
    let disconnected = { revoked: true };
    if (provider === "aipass") {
      try {
        disconnected = await aiPass.disconnect();
      } catch (error) {
        // Do not remove the runtime marker or claim a disconnect when native secure storage could
        // not clear the bearer credentials. A revocation-network failure is represented by
        // `{ revoked:false }` only after local clearing has definitely succeeded.
        return aiErr(c, error, provider);
      }
    }
    if (cfg.ai?.providers) delete cfg.ai.providers[provider];
    if (provider !== "aipass") {
      await deleteSecret(aiKeyName(provider)); // drop the key from the OS keychain too
    }
    if (cfg.ai && cfg.ai.defaultProvider === provider) {
      cfg.ai.defaultProvider = AI_PROVIDERS.find(
        (p) => providerConfigured(cfg, p) && !!resolveModel(cfg, p),
      );
    }
    saveConfig(cfg);
    return c.json({
      ...redactAi(cfg),
      ...(provider === "aipass" ? { aipassRevoked: disconnected.revoked } : {}),
    });
  });

  // Draft a commit message from the repo's diff using the default (or a chosen) provider.
  app.post("/api/repos/:id/commit-message", async (c) => {
    const id = requireId(c);
    if (id instanceof Response) return id;
    const p = await parseBody(c, CommitMessageSchema);
    if (!p.ok) return p.res;
    const guest = effectiveGuest(c, cfg);
    if (guest && cfg.ai?.commitEnabled === false) {
      return jsonError(c, "FORBIDDEN", "AI commit generation is disabled by the owner", 403);
    }
    // Share guests may spend only the provider/model the owner selected as default. Provider
    // overrides and provider identity remain owner-only.
    const requested = guest || p.data.provider == null ? undefined : (p.data.provider as AiProviderId);
    const provider = requested ?? effectiveDefaultProvider(cfg);
    if (!provider) return jsonError(c, "NO_AI_PROVIDER", "no AI provider configured");
    if (!providerConfigured(cfg, provider)) {
      return jsonError(c, "NO_AI_PROVIDER", `${provider} is not configured`);
    }
    const apiKey = provider === "aipass" ? "" : resolveApiKey(cfg, provider)!;
    const model = resolveModel(cfg, provider);
    if (!model) return jsonError(c, "NO_MODEL", `pick a model for ${provider} in Settings`);

    // With `paths`, draft from only those files (smart-commit per-group regenerate); else the
    // whole working tree (the normal "Generate" button). Both honor the owner's diff-detail dial.
    const msgDetail = cfg.ai?.diffDetail ?? DEFAULT_DIFF_DETAIL;
    const collected =
      p.data.paths?.length
        ? await collectRepoPathsDiff(id, p.data.paths, msgDetail)
        : await collectRepoDiff(id, msgDetail);
    if (!collected.ok) {
      const status: ContentfulStatusCode =
        collected.code === "NOT_FOUND" ? 404 : collected.code === "NOTHING_TO_COMMIT" ? 409 : 400;
      return c.json(collected, status);
    }
    const admission = enterGuestAi(c);
    if (admission instanceof Response) return admission;
    try {
      const message = await generateCommitMessage(
        provider,
        apiKey,
        model,
        collected.diff!,
        cfg.ai?.style ?? "conventional",
        undefined,
        collected.files ?? 0, // anchors the body's bullet floor to the real file count
        {
          signal: c.req.raw.signal,
          ...(provider === "aipass"
            ? {
                streamCompletion: (
                  body: Record<string, unknown>,
                  options: { signal?: AbortSignal; timeoutMs: number },
                ) => aiPass.streamCompletion(body, options),
              }
            : {}),
        },
      );
      return c.json(guest ? { ok: true, message } : { ok: true, message, provider, model });
    } catch (e) {
      return aiErr(c, e, provider);
    } finally {
      admission?.();
    }
  });

  // Propose a multi-commit plan from the repo's working tree (read-only — commits NOTHING).
  // On an AI failure other than a bad key we fall back to a deterministic grouping so Smart
  // Commit always yields an editable plan; a rejected key surfaces so the owner can fix it.
  app.post("/api/repos/:id/commit-plan", async (c) => {
    const id = requireId(c);
    if (id instanceof Response) return id;
    const p = await parseBody(c, CommitPlanSchema);
    if (!p.ok) return p.res;
    const guest = effectiveGuest(c, cfg);
    if (guest && cfg.ai?.commitEnabled === false) {
      return jsonError(c, "FORBIDDEN", "AI commit generation is disabled by the owner", 403);
    }
    const requested = guest || p.data.provider == null ? undefined : (p.data.provider as AiProviderId);
    const provider = requested ?? effectiveDefaultProvider(cfg);
    if (!provider) return jsonError(c, "NO_AI_PROVIDER", "no AI provider configured");
    if (!providerConfigured(cfg, provider)) {
      return jsonError(c, "NO_AI_PROVIDER", `${provider} is not configured`);
    }
    const apiKey = provider === "aipass" ? "" : resolveApiKey(cfg, provider)!;
    const model = resolveModel(cfg, provider);
    if (!model) return jsonError(c, "NO_MODEL", `pick a model for ${provider} in Settings`);

    // Empty selection means "nothing checked" → plan the whole tree, so an empty array is
    // treated the same as omitting `paths` entirely (never an accidental empty-scope plan).
    const collected = await planCommitInput(
      id,
      p.data.paths?.length ? p.data.paths : undefined,
      cfg.ai?.diffDetail ?? DEFAULT_DIFF_DETAIL,
    );
    if (!collected.ok) {
      const status: ContentfulStatusCode =
        collected.code === "NOT_FOUND" ? 404 : collected.code === "NOTHING_TO_COMMIT" ? 409 : 400;
      return c.json(collected, status);
    }
    const style = cfg.ai?.style ?? "conventional";
    const admission = enterGuestAi(c);
    if (admission instanceof Response) return admission;
    try {
      const plan = await generateCommitPlan(
        provider,
        apiKey,
        model,
        collected.input!,
        style,
        undefined,
        {
          signal: c.req.raw.signal,
          ...(provider === "aipass"
            ? {
                retryMalformed: false,
                streamCompletion: (
                  body: Record<string, unknown>,
                  options: { signal?: AbortSignal; timeoutMs: number },
                ) => aiPass.streamCompletion(body, options),
              }
            : {}),
        },
      );
      return c.json(guest ? { ok: true, plan } : { ok: true, plan, provider, model });
    } catch (e) {
      // A bad/rejected key is worth surfacing (the owner must fix it); anything else
      // (provider down, rate limit, garbage response) still falls back to the deterministic
      // plan so Smart Commit never dead-ends — but the REASON rides along. It used to be
      // dropped here, which made a rate-limited request (where the model never ran at all)
      // render as "AI couldn't structure this" — a wrong answer to a question the owner can
      // actually act on ("your daily token cap is spent; retry at X / switch provider").
      if (
        (e instanceof AiError && e.code === "AI_AUTH_FAILED") ||
        (e instanceof AiPassError &&
          (e.code === "AI_AUTH_FAILED" || e.code === "NOT_CONFIGURED"))
      ) {
        return aiErr(c, e, provider);
      }
      const reason =
        e instanceof AiError
          ? { code: e.code, message: e.message }
          : e instanceof AiPassError && e.code !== "NOT_CONFIGURED"
            ? { code: e.code, message: e.message }
          : { code: "AI_ERROR" as const, message: e instanceof Error ? e.message : String(e) };
      const plan = heuristicPlan(collected.input!, reason);
      return c.json(
        guest
          ? { ok: true, plan, fallback: true }
          : { ok: true, plan, provider, model, fallback: true },
      );
    } finally {
      admission?.();
    }
  });
}
