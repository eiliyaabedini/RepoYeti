/**
 * Non-secret, persistent fail-closed marker for an unusable AI Pass credential generation.
 *
 * The bearer tokens remain exclusively in native secure storage. This small marker only prevents
 * a retained token from becoming usable again after a daemon restart when the credential backend
 * could neither delete nor overwrite it. A successful new OAuth connection removes the marker.
 */
import { existsSync, mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export interface AiPassBlockStore {
  get(): boolean;
  set(blocked: boolean): boolean;
}

const configDir = process.env.REPOYETI_HOME ?? join(homedir(), ".repoyeti");
const markerPath = join(configDir, "aipass.blocked");
const pendingPath = join(configDir, `aipass.blocked.${process.pid}.tmp`);

export const nativeAiPassBlockStore: AiPassBlockStore = {
  get(): boolean {
    try {
      return existsSync(markerPath);
    } catch {
      return true;
    }
  },
  set(blocked: boolean): boolean {
    try {
      if (blocked) {
        mkdirSync(configDir, { recursive: true, mode: 0o700 });
        writeFileSync(pendingPath, "blocked\n", { encoding: "utf8", mode: 0o600 });
        renameSync(pendingPath, markerPath);
        return existsSync(markerPath);
      }
      rmSync(markerPath, { force: true });
      return !existsSync(markerPath);
    } catch {
      try {
        rmSync(pendingPath, { force: true });
      } catch {
        /* best-effort cleanup of a non-secret temporary marker */
      }
      return false;
    }
  },
};
