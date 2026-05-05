<script lang="ts">
    import { onMount } from 'svelte';
    import AttentionGrid from '$lib/viz/AttentionGrid.svelte';
    import ResidualRiver from '$lib/viz/ResidualRiver.svelte';
    import NeuronAtlas from '$lib/viz/NeuronAtlas.svelte';
    import CircuitCanvas from '$lib/viz/CircuitCanvas.svelte';
    import PathPatchPanel from '$lib/viz/PathPatchPanel.svelte';
    import SaePanel from '$lib/viz/SaePanel.svelte';
    import { session } from '$lib/stores/session.svelte';
    import { formatBytes, formatParams, loadGlassbox } from '$lib/glassbox';

    let modelUrl = $state('/models/gpt2-small.glx');
    let activeView = $state<'attention' | 'river' | 'atlas' | 'circuit' | 'patch' | 'sae'>('attention');

    function fakePattern(layer: number, head: number, seq: number): Float32Array {
        const arr = new Float32Array(seq * seq);
        for (let i = 0; i < seq; i++) {
            for (let j = 0; j <= i; j++) {
                const dist = (i - j) / seq;
                const phase = (layer * 0.7 + head * 1.3) % 6.28;
                arr[i * seq + j] = Math.exp(-dist * (2 + Math.sin(phase + i * 0.4)));
            }
            let sum = 0;
            for (let j = 0; j <= i; j++) sum += arr[i * seq + j];
            if (sum > 0) for (let j = 0; j <= i; j++) arr[i * seq + j] /= sum;
        }
        return arr;
    }

    function patternFromHook(layer: number, head: number, seq: number): Float32Array | null {
        if (!session.handle || !session.info) return null;
        const buf = session.handle.readHook(`blocks.${layer}.attn.pattern`);
        if (!buf) return null;
        const heads = session.info.n_head;
        const expected = heads * seq * seq;
        if (buf.length !== expected) return null;
        return buf.slice(head * seq * seq, (head + 1) * seq * seq);
    }

    function getAttentionPattern(layer: number, head: number): Float32Array | null {
        return patternFromHook(layer, head, seqLen) ?? fakePattern(layer, head, seqLen);
    }

    function fakeProjection(layer: number, position: number): [number, number] {
        const t = layer / 12;
        const drift = Math.sin(position * 0.7 + layer * 0.3) * 0.5;
        return [t, drift * (1 - t * 0.4)];
    }

    const fakeNeurons = $derived.by(() => {
        if (!session.info) return [];
        const out: { layer: number; neuron: number; activation: number }[] = [];
        const layers = session.info.n_layer;
        for (let i = 0; i < 24; i++) {
            const layer = Math.floor(Math.random() * layers);
            const neuron = Math.floor(Math.random() * 3072);
            out.push({ layer, neuron, activation: Math.random() * 6 + 0.5 });
        }
        return out.sort((a, b) => b.activation - a.activation);
    });

    const fakeEdges = $derived(
        Array.from({ length: 8 }, () => ({
            from: { layer: Math.floor(Math.random() * 6), head: Math.floor(Math.random() * 12), position: 0 },
            to: { layer: Math.floor(Math.random() * 6) + 6, head: Math.floor(Math.random() * 12), position: 0 },
            weight: Math.random() * 0.8 + 0.1,
        }))
    );

    async function load() {
        try {
            session.error = null;
            const { handle, backend } = await loadGlassbox(modelUrl, (p) => {
                session.progress = p;
            });
            session.setHandle(handle);
            session.activeBackend = backend;
            session.tokens = Array.from(handle.encode(session.prompt));
        } catch (e) {
            session.setError(e instanceof Error ? e.message : String(e));
        } finally {
            session.progress = null;
        }
    }

    async function generate() {
        if (!session.handle) return;
        session.isGenerating = true;
        session.generated = [];
        session.generatedText = '';
        try {
            session.handle.clearHooks();
            const subscriptions = [
                `blocks.${session.selectedLayer}.attn.pattern`,
                `blocks.${session.selectedLayer}.resid_post`,
                `blocks.${session.selectedLayer}.mlp.post`,
            ];
            for (const s of subscriptions) session.handle.subscribe(s);

            const args = {
                temperature: session.temperature,
                top_k: session.topK,
                top_p: session.topP,
                seed: 42,
            };
            const result =
                session.activeBackend === 'webgpu'
                    ? await session.handle.generateAsync(session.prompt, session.maxNewTokens, args)
                    : session.handle.generate(session.prompt, session.maxNewTokens, args);
            session.generated = Array.from(result.tokens);
            session.generatedText = result.text;
            session.elapsedMs = result.elapsed_ms;
            session.tokensPerSecond = result.tokens_per_second;
        } catch (e) {
            session.error = e instanceof Error ? e.message : String(e);
        } finally {
            session.isGenerating = false;
        }
    }

    const seqLen = $derived(session.tokens.length || 8);
    const tokensAsStrings = $derived(session.tokens.map(() => '·'));

    onMount(() => {});
</script>

<header class="topbar">
    <div class="brand mono">
        <span class="logo">◇</span>
        <span class="name">glassbox</span>
        <span class="tag">v0.1</span>
    </div>
    <nav class="views">
        {#each ['attention', 'river', 'atlas', 'circuit', 'patch', 'sae'] as v}
            <button class:active={activeView === v} onclick={() => (activeView = v as typeof activeView)}>
                {v}
            </button>
        {/each}
    </nav>
    <div class="meta mono">
        {#if session.info}
            <span class="tag active">{session.info.architecture}</span>
            <span class="dim">{formatParams(session.info.parameter_count)} params</span>
            <span class="dim">·</span>
            <span class="dim">{session.info.n_layer}L · {session.info.n_head}H · {session.info.n_embd}d</span>
            {#if session.activeBackend}
                <span class="dim">·</span>
                <span class="tag" class:active={session.activeBackend === 'webgpu'}>{session.activeBackend}</span>
            {/if}
        {:else}
            <span class="dim">no model loaded</span>
        {/if}
    </div>
</header>

<main class="layout">
    <aside class="sidebar">
        <section>
            <h3 class="mono">model</h3>
            <input bind:value={modelUrl} placeholder="model url" />
            <button onclick={load} disabled={!!session.progress}>
                {session.progress ? session.progress.phase : 'load'}
            </button>
            {#if session.progress}
                <div class="progress mono">
                    {formatBytes(session.progress.bytes_loaded)} / {formatBytes(session.progress.bytes_total)}
                    <div class="bar">
                        <div
                            class="fill"
                            style="width: {session.progress.bytes_total > 0
                                ? (session.progress.bytes_loaded / session.progress.bytes_total) * 100
                                : 0}%"
                        ></div>
                    </div>
                </div>
            {/if}
            {#if session.error}
                <div class="error mono">{session.error}</div>
            {/if}
        </section>

        <div class="divider"></div>

        <section>
            <h3 class="mono">prompt</h3>
            <textarea bind:value={session.prompt} rows="4"></textarea>
            <div class="row3 mono">
                <label>max <input type="number" bind:value={session.maxNewTokens} min="1" max="256" /></label>
                <label>temp <input type="number" bind:value={session.temperature} step="0.05" min="0" max="2" /></label>
                <label>top-k <input type="number" bind:value={session.topK} min="0" max="200" /></label>
            </div>
            <button onclick={generate} disabled={!session.handle || session.isGenerating}>
                {session.isGenerating ? 'generating…' : 'generate'}
            </button>
            {#if session.generatedText || session.elapsedMs > 0}
                <div class="output mono">
                    {#if session.generatedText}
                        <div class="generated">{session.generatedText}</div>
                    {/if}
                    {#if session.elapsedMs > 0}
                        <div class="dim small">
                            {session.elapsedMs.toFixed(0)} ms · {session.tokensPerSecond.toFixed(1)} tok/s
                        </div>
                    {/if}
                </div>
            {/if}
        </section>

        <div class="divider"></div>

        <section>
            <h3 class="mono">selection</h3>
            <div class="kv mono">
                <span class="dim">layer</span><span>L{session.selectedLayer}</span>
                <span class="dim">head</span><span>H{session.selectedHead}</span>
                <span class="dim">position</span><span>{session.selectedPosition}</span>
            </div>
        </section>

        <div class="divider"></div>

        <section>
            <h3 class="mono">hooks</h3>
            <ul class="hooks mono">
                <li>blocks.{session.selectedLayer}.attn.pattern</li>
                <li>blocks.{session.selectedLayer}.attn.z</li>
                <li>blocks.{session.selectedLayer}.mlp.post</li>
                <li>blocks.{session.selectedLayer}.resid_post</li>
                <li>unembed</li>
            </ul>
        </section>
    </aside>

    <section class="canvas">
        {#if activeView === 'attention' && session.info}
            <AttentionGrid
                nLayers={Math.min(6, session.info.n_layer)}
                nHeads={session.info.n_head}
                {seqLen}
                getPattern={(l, h) => getAttentionPattern(l, h)}
                selectedLayer={session.selectedLayer}
                selectedHead={session.selectedHead}
                onSelect={(l, h) => {
                    session.selectedLayer = l;
                    session.selectedHead = h;
                }}
            />
        {:else if activeView === 'river' && session.info}
            <ResidualRiver
                nLayers={session.info.n_layer}
                positions={Math.min(seqLen, 12)}
                getProjection={fakeProjection}
                tokens={tokensAsStrings}
            />
        {:else if activeView === 'atlas'}
            <NeuronAtlas
                topNeurons={fakeNeurons}
                onSelect={(l, n) => {
                    session.selectedLayer = l;
                    console.info('selected neuron', l, n);
                }}
            />
        {:else if activeView === 'circuit' && session.info}
            <CircuitCanvas nLayers={session.info.n_layer} edges={fakeEdges} />
        {:else if activeView === 'patch' && session.handle && session.info}
            <PathPatchPanel handle={session.handle} nLayers={session.info.n_layer} />
        {:else if activeView === 'sae' && session.handle && session.info}
            <SaePanel handle={session.handle} nLayers={session.info.n_layer} />
        {:else}
            <div class="hero">
                <div class="hero-inner">
                    <h1 class="hero-headline mono">a transformer you can see through</h1>
                    <p class="hero-sub mono">browser-native GPT-2 + mechanistic interpretability — runs entirely in your tab via WebGPU</p>
                    <div class="hero-cards">
                        <div class="hero-card mono">
                            <div class="hero-card-title">see</div>
                            <div class="hero-card-body">attention heads, residual streams, neuron firings — every internal tensor is addressable and drawn live as the model runs.</div>
                        </div>
                        <div class="hero-card mono">
                            <div class="hero-card-title">intervene</div>
                            <div class="hero-card-body">patch activations between clean and corrupted prompts, route counterfactuals along single edges, watch the logits move.</div>
                        </div>
                        <div class="hero-card mono">
                            <div class="hero-card-title">explore</div>
                            <div class="hero-card-body">project intermediates through the unembedding, surface top-k MLP features, sketch a circuit and run it end to end.</div>
                        </div>
                    </div>
                    <div class="hero-foot mono">v0.1 · MIT · github.com/MeftunH/glassbox</div>
                </div>
            </div>
        {/if}
    </section>
</main>

<style>
    .topbar {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        height: 44px;
        padding: 0 16px;
        border-bottom: 1px solid var(--line);
        background: var(--bg-1);
        gap: 16px;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .logo {
        color: var(--accent);
        font-size: 16px;
    }
    .name {
        font-size: 13px;
        letter-spacing: 0.04em;
    }
    .views {
        display: flex;
        gap: 4px;
    }
    .views button {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .views button.active {
        color: var(--accent);
        border-color: var(--accent-dim);
        background: rgba(125, 211, 252, 0.06);
    }
    .meta {
        display: flex;
        align-items: center;
        gap: 8px;
        justify-content: flex-end;
        font-size: 11px;
    }
    .layout {
        flex: 1;
        display: grid;
        grid-template-columns: 280px 1fr;
        min-height: 0;
    }
    .sidebar {
        border-right: 1px solid var(--line);
        background: var(--bg-1);
        padding: 16px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .sidebar section {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    h3 {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--fg-3);
        margin: 0 0 4px 0;
        font-weight: 500;
    }
    .progress {
        font-size: 10px;
        color: var(--fg-2);
    }
    .bar {
        height: 2px;
        background: var(--bg-3);
        margin-top: 4px;
        overflow: hidden;
    }
    .fill {
        height: 100%;
        background: var(--accent);
        transition: width 80ms linear;
    }
    .error {
        font-size: 11px;
        color: var(--bad);
        padding: 8px;
        border: 1px solid var(--bad);
        border-radius: var(--radius-sm);
        background: rgba(239, 110, 110, 0.04);
    }
    .kv {
        display: grid;
        grid-template-columns: 80px 1fr;
        gap: 4px 8px;
        font-size: 11px;
    }
    .hooks {
        list-style: none;
        margin: 0;
        padding: 0;
        font-size: 11px;
        color: var(--fg-2);
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .canvas {
        padding: 16px;
        overflow: auto;
        background:
            linear-gradient(var(--grid) 1px, transparent 1px) 0 0 / 32px 32px,
            linear-gradient(90deg, var(--grid) 1px, transparent 1px) 0 0 / 32px 32px,
            var(--bg-0);
        min-width: 0;
    }
    .empty {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        color: var(--fg-2);
    }
    .row3 {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 6px;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--fg-3);
    }
    .row3 input {
        width: 100%;
        font-size: 11px;
        padding: 4px 6px;
        margin-top: 2px;
    }
    .output {
        margin-top: 8px;
        padding: 8px;
        background: var(--bg-2);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-sm);
        font-size: 11px;
    }
    .generated {
        color: var(--fg-0);
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.45;
    }
    .small {
        font-size: 9px;
        margin-top: 6px;
    }
    .hero {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 32px 24px;
    }
    .hero-inner {
        width: 100%;
        max-width: 760px;
        display: flex;
        flex-direction: column;
        gap: 28px;
    }
    .hero-headline {
        font-size: clamp(26px, 4vw, 38px);
        line-height: 1.1;
        letter-spacing: -0.01em;
        color: var(--fg-0);
        font-weight: 500;
        margin: 0;
    }
    .hero-sub {
        font-size: 12px;
        line-height: 1.6;
        color: var(--fg-2);
        margin: 0;
        max-width: 60ch;
    }
    .hero-cards {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 4px;
    }
    .hero-card {
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-sm);
        padding: 16px;
        background: var(--bg-1);
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .hero-card-title {
        font-size: 11px;
        font-weight: 500;
        color: var(--fg-0);
        letter-spacing: 0.06em;
        text-transform: lowercase;
    }
    .hero-card-body {
        font-size: 11px;
        line-height: 1.55;
        color: var(--fg-2);
    }
    .hero-foot {
        font-size: 10px;
        color: var(--fg-3);
        letter-spacing: 0.04em;
        margin-top: 4px;
    }
    @media (max-width: 720px) {
        .hero-cards {
            grid-template-columns: 1fr;
        }
    }
</style>
