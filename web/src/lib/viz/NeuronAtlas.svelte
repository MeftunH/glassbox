<script lang="ts">
    interface Props {
        topNeurons: { layer: number; neuron: number; activation: number }[];
        onSelect?: (layer: number, neuron: number) => void;
    }

    let { topNeurons, onSelect }: Props = $props();

    const max = $derived(topNeurons.reduce((a, b) => Math.max(a, b.activation), 0));
</script>

<div class="atlas">
    <header class="head">
        <span class="mono dim">top neurons by activation</span>
        <span class="mono dim small">L · N · activation</span>
    </header>
    <div class="rows">
        {#each topNeurons as n}
            {@const w = max > 0 ? (n.activation / max) * 100 : 0}
            <button class="row" onclick={() => onSelect?.(n.layer, n.neuron)}>
                <span class="mono cell-l">L{n.layer}</span>
                <span class="mono cell-n">{n.neuron.toString().padStart(4, '0')}</span>
                <div class="bar">
                    <div class="bar-fill" style="width: {w}%"></div>
                </div>
                <span class="mono cell-v">{n.activation.toFixed(3)}</span>
            </button>
        {/each}
        {#if topNeurons.length === 0}
            <div class="empty mono dim">no activations recorded yet</div>
        {/if}
    </div>
</div>

<style>
    .atlas {
        display: flex;
        flex-direction: column;
        background: var(--bg-1);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-md);
        height: 100%;
        min-height: 200px;
        overflow: hidden;
    }
    .head {
        display: flex;
        justify-content: space-between;
        padding: 8px 12px;
        border-bottom: 1px solid var(--line-soft);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .small {
        font-size: 9px;
    }
    .rows {
        flex: 1;
        overflow-y: auto;
        padding: 4px 8px;
    }
    .row {
        display: grid;
        grid-template-columns: 28px 48px 1fr 56px;
        gap: 8px;
        align-items: center;
        width: 100%;
        background: transparent;
        border: 1px solid transparent;
        border-radius: var(--radius-sm);
        padding: 4px 6px;
        font-size: 11px;
        color: var(--fg-1);
    }
    .row:hover {
        background: var(--bg-2);
        border-color: var(--line-soft);
        color: var(--fg-0);
    }
    .cell-l {
        color: var(--fg-3);
    }
    .cell-n {
        color: var(--fg-1);
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
    }
    .empty {
        padding: 24px;
        text-align: center;
        font-size: 11px;
    }
</style>
