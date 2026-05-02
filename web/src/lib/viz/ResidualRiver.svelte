<script lang="ts">
    import { onMount } from 'svelte';
    import { line, curveMonotoneX } from 'd3-shape';
    import { scaleLinear } from 'd3-scale';

    interface Props {
        nLayers: number;
        positions: number;
        getProjection: (layer: number, position: number) => [number, number] | null;
        tokens: string[];
    }

    let { nLayers, positions, getProjection, tokens }: Props = $props();

    let width = $state(720);
    let height = $state(280);
    let container: HTMLDivElement | null = $state(null);

    onMount(() => {
        if (!container) return;
        const ro = new ResizeObserver((entries) => {
            for (const e of entries) {
                width = Math.max(320, e.contentRect.width);
                height = Math.max(180, e.contentRect.height);
            }
        });
        ro.observe(container);
        return () => ro.disconnect();
    });

    const xScale = $derived(scaleLinear().domain([0, nLayers - 1]).range([40, width - 16]));
    const yScale = $derived(scaleLinear().domain([-1, 1]).range([height - 24, 16]));

    const lineGen = $derived(
        line<{ x: number; y: number }>()
            .x((d) => d.x)
            .y((d) => d.y)
            .curve(curveMonotoneX)
    );

    const trajectories = $derived.by(() => {
        const out: { tokenIdx: number; path: string; tail: { x: number; y: number } | null }[] = [];
        for (let p = 0; p < positions; p++) {
            const points: { x: number; y: number }[] = [];
            for (let l = 0; l < nLayers; l++) {
                const proj = getProjection(l, p);
                if (!proj) continue;
                points.push({ x: xScale(l), y: yScale(proj[1]) });
            }
            if (points.length < 2) continue;
            out.push({
                tokenIdx: p,
                path: lineGen(points) ?? '',
                tail: points[points.length - 1] ?? null,
            });
        }
        return out;
    });
</script>

<div bind:this={container} class="river">
    <svg {width} {height}>
        <defs>
            <linearGradient id="trajectoryGradient" x1="0" x2="1" y1="0" y2="0">
                <stop offset="0%" stop-color="#38617a" stop-opacity="0" />
                <stop offset="20%" stop-color="#7dd3fc" stop-opacity="0.6" />
                <stop offset="100%" stop-color="#7dd3fc" stop-opacity="0.95" />
            </linearGradient>
        </defs>

        {#each { length: nLayers } as _, l}
            <line
                x1={xScale(l)}
                x2={xScale(l)}
                y1={16}
                y2={height - 24}
                stroke="var(--line-soft)"
                stroke-width="1"
                stroke-dasharray="2 4"
            />
            <text
                x={xScale(l)}
                y={height - 8}
                text-anchor="middle"
                font-family="var(--font-mono)"
                font-size="10"
                fill="var(--fg-3)"
            >
                L{l}
            </text>
        {/each}

        {#each trajectories as t}
            <path
                d={t.path}
                stroke="url(#trajectoryGradient)"
                stroke-width="1.25"
                fill="none"
                opacity="0.85"
            />
            {#if t.tail}
                <circle cx={t.tail.x} cy={t.tail.y} r="2.5" fill="var(--accent)" />
                <text
                    x={t.tail.x + 6}
                    y={t.tail.y + 3}
                    font-family="var(--font-mono)"
                    font-size="9"
                    fill="var(--fg-1)"
                >
                    {tokens[t.tokenIdx] ?? ''}
                </text>
            {/if}
        {/each}
    </svg>
</div>

<style>
    .river {
        width: 100%;
        height: 100%;
        min-height: 200px;
        background: var(--bg-1);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-md);
    }
    svg {
        display: block;
    }
</style>
