<script lang="ts">
    import type { GlassboxHandle, PathPatchOut } from '$lib/glassbox';

    interface Props {
        handle: GlassboxHandle;
        nLayers: number;
    }

    let { handle, nLayers }: Props = $props();

    let cleanPrompt = $state('When Mary went to the store, John gave a book to');
    let corruptPrompt = $state('When John went to the store, Mary gave a book to');
    let senderLayer = $state(5);
    let receiverLayer = $state(9);
    let result = $state<PathPatchOut | null>(null);
    let running = $state(false);
    let error = $state<string | null>(null);

    async function run() {
        running = true;
        error = null;
        try {
            const out = handle.runPathPatch({
                clean_prompt: cleanPrompt,
                corrupt_prompt: corruptPrompt,
                sender_hook: `blocks.${senderLayer}.resid_post`,
                receiver_hooks: [`blocks.${receiverLayer}.resid_post`],
                target_token: null,
            });
            result = out;
        } catch (e) {
            error = e instanceof Error ? e.message : String(e);
        } finally {
            running = false;
        }
    }

    const recoveryPercent = $derived(result ? Math.round(result.recovery * 100) : 0);
</script>

<div class="panel">
    <header>
        <h3 class="mono">path patching</h3>
        <span class="dim mono small">clean → corrupt + sender patch · measure logit recovery</span>
    </header>

    <div class="form">
        <label>
            <span class="mono dim">clean</span>
            <textarea bind:value={cleanPrompt} rows="2"></textarea>
        </label>
        <label>
            <span class="mono dim">corrupt</span>
            <textarea bind:value={corruptPrompt} rows="2"></textarea>
        </label>
        <div class="row">
            <label>
                <span class="mono dim">sender layer</span>
                <input type="number" min="0" max={nLayers - 1} bind:value={senderLayer} />
            </label>
            <label>
                <span class="mono dim">receiver layer</span>
                <input type="number" min="0" max={nLayers - 1} bind:value={receiverLayer} />
            </label>
        </div>
        <button onclick={run} disabled={running}>
            {running ? 'running…' : 'run patch'}
        </button>
        {#if error}<div class="error mono">{error}</div>{/if}
    </div>

    {#if result}
        <div class="results mono">
            <div class="logit-row">
                <span class="dim">clean</span>
                <span class="value">{result.clean_logit.toFixed(3)}</span>
            </div>
            <div class="logit-row">
                <span class="dim">corrupt</span>
                <span class="value">{result.corrupt_logit.toFixed(3)}</span>
            </div>
            <div class="logit-row">
                <span class="dim">patched</span>
                <span class="value accent">{result.patched_logit.toFixed(3)}</span>
            </div>
            <div class="recovery">
                <div class="recovery-label">
                    <span class="dim">recovery</span>
                    <span class="value accent">{recoveryPercent}%</span>
                </div>
                <div class="bar">
                    <div class="fill" style="width: {Math.max(0, Math.min(100, recoveryPercent))}%"></div>
                </div>
                <div class="dim small">{result.elapsed_ms.toFixed(0)} ms</div>
            </div>
        </div>
    {/if}
</div>

<style>
    .panel {
        display: flex;
        flex-direction: column;
        gap: 12px;
        background: var(--bg-1);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-md);
        padding: 16px;
    }
    header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
    }
    h3 {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--fg-3);
        margin: 0;
        font-weight: 500;
    }
    .small {
        font-size: 10px;
    }
    .form {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .form label {
        display: flex;
        flex-direction: column;
        gap: 3px;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .error {
        color: var(--bad);
        padding: 6px 8px;
        border: 1px solid var(--bad);
        border-radius: var(--radius-sm);
        font-size: 10px;
    }
    .results {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 10px;
        background: var(--bg-2);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-sm);
    }
    .logit-row {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
    }
    .value {
        color: var(--fg-1);
    }
    .accent {
        color: var(--accent);
    }
    .recovery {
        margin-top: 4px;
        padding-top: 8px;
        border-top: 1px solid var(--line-soft);
    }
    .recovery-label {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        margin-bottom: 6px;
    }
    .bar {
        height: 4px;
        background: var(--bg-3);
        border-radius: 2px;
        overflow: hidden;
    }
    .fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-dim), var(--accent));
        transition: width 200ms ease;
    }
</style>
