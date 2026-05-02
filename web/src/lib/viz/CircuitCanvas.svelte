<script lang="ts">
    interface Edge {
        from: { layer: number; head: number; position: number };
        to: { layer: number; head: number; position: number };
        weight: number;
    }

    interface Props {
        nLayers: number;
        edges: Edge[];
    }

    let { nLayers, edges }: Props = $props();

    const width = 720;
    const height = 280;

    function nodeX(layer: number) {
        return 40 + (layer / Math.max(1, nLayers - 1)) * (width - 80);
    }
    function nodeY(head: number, position: number) {
        const lane = (head * 7 + position * 11) % 9;
        return 40 + lane * 26;
    }
</script>

<div class="circuit">
    <svg {width} {height} viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">
        <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent)" />
            </marker>
        </defs>

        {#each { length: nLayers } as _, l}
            <line x1={nodeX(l)} x2={nodeX(l)} y1={20} y2={height - 20} stroke="var(--line-soft)" stroke-dasharray="2 3" />
            <text x={nodeX(l)} y={14} text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="var(--fg-3)">
                L{l}
            </text>
        {/each}

        {#each edges as e}
            {@const x1 = nodeX(e.from.layer)}
            {@const y1 = nodeY(e.from.head, e.from.position)}
            {@const x2 = nodeX(e.to.layer)}
            {@const y2 = nodeY(e.to.head, e.to.position)}
            <path
                d="M {x1} {y1} C {(x1 + x2) / 2} {y1}, {(x1 + x2) / 2} {y2}, {x2} {y2}"
                stroke="var(--accent)"
                stroke-width={Math.max(0.5, Math.min(3, Math.abs(e.weight) * 4))}
                fill="none"
                opacity={0.4 + Math.min(0.5, Math.abs(e.weight))}
                marker-end="url(#arrow)"
            />
        {/each}

        {#each edges as e}
            <circle cx={nodeX(e.from.layer)} cy={nodeY(e.from.head, e.from.position)} r="3" fill="var(--accent)" />
            <circle cx={nodeX(e.to.layer)} cy={nodeY(e.to.head, e.to.position)} r="3" fill="var(--accent)" />
        {/each}

        {#if edges.length === 0}
            <text x={width / 2} y={height / 2} text-anchor="middle" font-family="var(--font-mono)" font-size="11" fill="var(--fg-3)">
                add edges to start a circuit
            </text>
        {/if}
    </svg>
</div>

<style>
    .circuit {
        background: var(--bg-1);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-md);
        height: 100%;
        min-height: 280px;
    }
    svg {
        display: block;
        width: 100%;
        height: 100%;
    }
</style>
