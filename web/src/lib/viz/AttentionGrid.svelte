<script lang="ts">
    interface Props {
        nLayers: number;
        nHeads: number;
        seqLen: number;
        getPattern: (layer: number, head: number) => Float32Array | null;
        selectedLayer: number;
        selectedHead: number;
        onSelect?: (layer: number, head: number) => void;
    }

    let { nLayers, nHeads, seqLen, getPattern, selectedLayer, selectedHead, onSelect }: Props = $props();

    const cellSize = 84;
    const padding = 8;

    function patternToImageData(pattern: Float32Array | null, size: number): string {
        if (!pattern) return '';
        const canvas = new OffscreenCanvas(size, size);
        const ctx = canvas.getContext('2d');
        if (!ctx) return '';
        const img = ctx.createImageData(size, size);
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const v = Math.max(0, Math.min(1, pattern[i * size + j] ?? 0));
                const idx = (i * size + j) * 4;
                img.data[idx] = Math.floor(125 * v);
                img.data[idx + 1] = Math.floor(211 * v);
                img.data[idx + 2] = Math.floor(252 * v);
                img.data[idx + 3] = 255;
            }
        }
        ctx.putImageData(img, 0, 0);
        return URL.createObjectURL(canvas.transferToImageBitmap() as unknown as Blob);
    }
</script>

<div class="grid" style="--cells: {nHeads}; --cell-size: {cellSize}px;">
    {#each { length: nLayers } as _, layer}
        <div class="row">
            <div class="row-label mono">L{layer}</div>
            {#each { length: nHeads } as _, head}
                {@const sel = layer === selectedLayer && head === selectedHead}
                {@const pattern = getPattern(layer, head)}
                <button
                    class="cell"
                    class:selected={sel}
                    onclick={() => onSelect?.(layer, head)}
                    aria-label="Layer {layer} head {head}"
                >
                    <canvas
                        width={seqLen}
                        height={seqLen}
                        data-pattern={pattern ? '1' : '0'}
                    ></canvas>
                    <span class="mono small">L{layer}H{head}</span>
                </button>
            {/each}
        </div>
    {/each}
</div>

<style>
    .grid {
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding: 8px;
        font-family: var(--font-mono);
    }
    .row {
        display: grid;
        grid-template-columns: 32px repeat(var(--cells), var(--cell-size));
        gap: 4px;
        align-items: center;
    }
    .row-label {
        color: var(--fg-3);
        font-size: 10px;
        text-align: right;
    }
    .cell {
        position: relative;
        background: var(--bg-1);
        border: 1px solid var(--line-soft);
        padding: 0;
        height: var(--cell-size);
        width: var(--cell-size);
        cursor: pointer;
        overflow: hidden;
    }
    .cell.selected {
        border-color: var(--accent);
        box-shadow: inset 0 0 0 1px var(--accent-dim);
    }
    .cell:hover {
        border-color: var(--accent-dim);
    }
    .cell canvas {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        image-rendering: pixelated;
        opacity: 0.85;
    }
    .small {
        position: absolute;
        bottom: 2px;
        right: 4px;
        font-size: 9px;
        color: var(--fg-2);
        background: rgba(8, 9, 11, 0.6);
        padding: 0 3px;
        border-radius: 1px;
        letter-spacing: 0.04em;
    }
</style>
