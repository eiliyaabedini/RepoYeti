import type { RepoYetiConfig } from "../config.ts";
import type { AiPassClient } from "../aipass.ts";
/** Shared state handed to every route module's register(). */
export interface Deps {
  cfg: RepoYetiConfig;
  aiPass: AiPassClient;
  requestShutdown?: () => void;
}
