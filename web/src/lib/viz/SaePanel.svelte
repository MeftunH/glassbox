<script lang="ts">
    import type { GlassboxHandle, SaeFeatureSpike } from '$lib/glassbox';

    interface Props {
        handle: GlassboxHandle;
        nLayers: number;
    }

    let { handle, nLayers }: Props = $props();

    let probeText = $state('When the model attends to');
    let layerIdx = $state(6);
    let hookKind = $state<'mlp.post' | 'resid_post' | 'attn.z'>('mlp.post');
    let features = $state<SaeFeatureSpike[]>([]);
    let running = $state(false);
    let loadError = $state<string | null>(null);
    let probeError = $state<string | null>(null);
    let loaded = $state<{ d_in: number; d_features: number } | null>(null);

    const currentHook = $derived(`blocks.${layerIdx}.${hookKind}`);
    const maxActivation = $derived(features.reduce((a, b) => Math.max(a, b.activation), 0));

    async function onFile(event: Event) {
        loadError = null;
        const target = event.target as HTMLInputElement;
        const file = target.files?.[0];
        if (!file) return;
        try {
            const text = await file.text();
            const json = JSON.parse(text) as {
                d_in: number;
                d_features: number;
                w_enc: number[];
                b_enc: number[];
                w_dec: number[];
                b_dec: number[];
            };
            handle.loadSae(
                'default',
                json.d_in,
                json.d_features,
                new Float32Array(json.w_enc),
                new Float32Array(json.b_enc),
                new Float32Array(json.w_dec),
                new Float32Array(json.b_dec)
            );
            loaded = { d_in: json.d_in, d_features: json.d_features };
        } catch (e) {
            loadError = e instanceof Error ? e.message : String(e);
            loaded = null;
        }
    }

    async function probe() {
        running = true;
        probeError = null;
        try {
            handle.clearHooks();
            handle.subscribe(currentHook);
            handle.forward(handle.encode(probeText));
            features = handle.encodeSaeFromHook('default', currentHook, 16);
        } catch (e) {
            probeError = e instanceof Error ? e.message : String(e);
        } finally {
            running = false;
        }
    }
</script>

<div class="panel">
    <header>
        <h3 class="mono">sparse autoencoder</h3>
        <span class="dim mono small">probe text → hook activation → top-k features</span>
    </header>

    <section class="block">
        <div class="block-head mono dim">load sae</div>
        <label class="file">
            <span class="mono dim">json file</span>
            <input type="file" accept="application/json,.json" onchange={onFile} />
        </label>
        {#if loaded}
            <div class="status mono">loaded: d_in={loaded.d_in}, features={loaded.d_features}</div>
        {/if}
        {#if loadError}<div class="error mono">{loadError}</div>{/if}
    </section>

    <section class="block">
        <div class="block-head mono dim">probe</div>
        <label>
            <span class="mono dim">probe text</span>
            <textarea bind:value={probeText} rows="2"></textarea>
        </label>
        <div class="row">
            <label>
                <span class="mono dim">layer</span>
                <input type="number" min="0" max={nLayers - 1} bind:value={layerIdx} />
            </label>
            <label>
                <span class="mono dim">hook</span>
                <select bind:value={hookKind}>
                    <option value="mlp.post">blocks.{layerIdx}.mlp.post</option>
                    <option value="resid_post">blocks.{layerIdx}.resid_post</option>
                    <option value="attn.z">blocks.{layerIdx}.attn.z</option>
                </select>
            </label>
        </div>
        <button onclick={probe} disabled={running}>
            {running ? 'probing…' : 'probe'}
        </button>
        {#if probeError}<div class="error mono">{probeError}</div>{/if}
    </section>

    <section class="block">
        <div class="block-head mono dim">features</div>
        <div class="rows">
            {#each features as f}
                {@const w = maxActivation > 0 ? (f.activation / maxActivation) * 100 : 0}
                <div class="row-feat">
                    <span class="mono cell-id">#{f.feature.toString().padStart(4, '0')}</span>
                    <div class="bar">
                        <div class="bar-fill" style="width: {w}%"></div>
                    </div>
                    <span class="mono cell-v">{f.activation.toFixed(3)}</span>
                </div>
            {/each}
            {#if features.length === 0}
                <div class="empty mono dim">no features captured yet</div>
            {/if}
        </div>
    </section>
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
    .block {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 10px;
        background: var(--bg-2);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-sm);
    }
    .block-head {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .block label {
        display: flex;
        flex-direction: column;
        gap: 3px;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .row {
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 8px;
    }
    .file input {
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--fg-1);
    }
    .status {
        font-size: 11px;
        color: var(--fg-2);
    }
    .error {
        color: var(--bad);
        padding: 6px 8px;
        border: 1px solid var(--bad);
        border-radius: var(--radius-sm);
        font-size: 10px;
    }
    .rows {
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    .row-feat {
        display: grid;
        grid-template-columns: 64px 1fr 64px;
        gap: 8px;
        align-items: center;
        padding: 4px 6px;
        font-size: 11px;
        color: var(--fg-1);
    }
    .cell-id {
        color: var(--fg-3);
    }
    .cell-v {
        color: var(--accent);
        text-align: right;
    }
    .bar {
        height: 6px;
        background: var(--bg-3);
        border-radius: 3px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-dim), var(--accent));
        transition: width 200ms ease;
    }
    .empty {
        padding: 16px;
        text-align: center;
        font-size: 11px;
    }
</style>
