import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AiProvidersSection from "@/components/settings/AiProvidersSection.vue";
import { i18n } from "@/i18n";
import { api } from "@/api";
import { useStore } from "@/store";

describe("AI Pass account connection", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    const store = useStore();
    store.aiCatalog = [
      {
        id: "aipass",
        label: "AI Pass",
        url: "aipass.one",
        accountConnection: true,
      },
      {
        id: "groq",
        label: "Groq",
        url: "console.groq.com/keys",
        keyPlaceholder: "gsk_…",
      },
    ];
  });

  it("offers Connect AI Pass without rendering an API-key field", async () => {
    const wrapper = mount(AiProvidersSection, {
      props: { open: true },
      global: { plugins: [i18n] },
    });
    const aiPassTrigger = wrapper
      .findAll("button")
      .find((button) => button.text().includes("AI Pass"));
    expect(aiPassTrigger).toBeDefined();
    await aiPassTrigger!.trigger("click");
    await flushPromises();

    const connect = wrapper.find('a[href="/api/ai/aipass/connect"]');
    expect(connect.exists()).toBe(true);
    expect(connect.text()).toContain("Connect AI Pass");
    expect(wrapper.find('input[aria-label="AI Pass API key"]').exists()).toBe(false);
    expect(wrapper.text()).toContain("shared AI Pass wallet");
  });

  it("threads cancellation from the browser request to generation", async () => {
    const call = vi
      .spyOn(api.ai, "commitMessage")
      .mockResolvedValue({
        ok: true,
        message: "feat: streamed",
        provider: "aipass",
        model: "live",
      });
    const signal = new AbortController().signal;

    await useStore().genCommitMessage("repo-1", undefined, undefined, signal);

    expect(call).toHaveBeenCalledWith("repo-1", undefined, undefined, signal);
  });
});
