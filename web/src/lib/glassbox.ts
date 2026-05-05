export interface ModelInfo {
    architecture: string;
    vocab_size: number;
    n_layer: number;
    n_head: number;
    n_embd: number;
    parameter_count: number;
}

export interface SamplingArgs {
    temperature: number;
    top_k: number | null;
    top_p: number | null;
    seed: number;
}

export interface GenerateOutput {
    tokens: Uint32Array;
    text: string;
    elapsed_ms: number;
    tokens_per_second: number;
}

export interface PathPatchArgs {
    clean_prompt: string;
    corrupt_prompt: string;
    sender_hook: string;
    receiver_hooks: string[];
    target_token: number | null;
}

export interface PathPatchOut {
    clean_logit: number;
    corrupt_logit: number;
    patched_logit: number;
    recovery: number;
    elapsed_ms: number;
}

export interface SaeFeatureSpike {
    feature: number;
    activation: number;
}

export interface GlassboxHandle {
    modelInfo(): ModelInfo;
    encode(text: string): Uint32Array;
    decode(ids: Uint32Array): string;
    subscribe(hook: string): void;
    unsubscribe(hook: string): void;
    readHook(hook: string): Float32Array | null;
    clearHooks(): void;
    forward(ids: Uint32Array): Float32Array;
    generate(prompt: string, maxNew: number, args: SamplingArgs): GenerateOutput;
    generateAsync(prompt: string, maxNew: number, args: SamplingArgs): Promise<GenerateOutput>;
    runPathPatch(args: PathPatchArgs): PathPatchOut;
    loadSae(
        key: string,
        d_in: number,
        d_features: number,
        w_enc: Float32Array,
        b_enc: Float32Array,
        w_dec: Float32Array,
        b_dec: Float32Array
    ): void;
    encodeSaeFromHook(saeKey: string, hook: string, topK: number): SaeFeatureSpike[];
    installPatch(hook: string, data: Float32Array, shape: Uint32Array): void;
    clearPatches(): void;
    backendName(): string;
}

export interface LoadProgress {
    phase: 'fetching' | 'parsing' | 'binding' | 'ready';
    bytes_loaded: number;
    bytes_total: number;
}

export type Backend = 'auto' | 'webgpu' | 'cpu';

export async function loadGlassbox(
    modelUrl: string,
    onProgress?: (p: LoadProgress) => void,
    backendPref: Backend = 'auto'
): Promise<{ handle: GlassboxHandle; backend: 'webgpu' | 'cpu' }> {
    onProgress?.({ phase: 'fetching', bytes_loaded: 0, bytes_total: 0 });

    const response = await fetch(modelUrl);
    if (!response.ok || !response.body) {
        throw new Error(`failed to fetch model: ${response.status} ${response.statusText}`);
    }

    const total = Number(response.headers.get('Content-Length') ?? '0');
    const chunks: Uint8Array[] = [];
    let received = 0;
    const reader = response.body.getReader();
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.byteLength;
        onProgress?.({ phase: 'fetching', bytes_loaded: received, bytes_total: total });
    }

    const blob = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
        blob.set(chunk, offset);
        offset += chunk.byteLength;
    }

    onProgress?.({ phase: 'parsing', bytes_loaded: received, bytes_total: total });

    const wasm = await import('./wasm/glassbox_wasm.js' as string);
    await wasm.default();

    onProgress?.({ phase: 'binding', bytes_loaded: received, bytes_total: total });

    const tryWebGpu =
        backendPref !== 'cpu' &&
        typeof navigator !== 'undefined' &&
        (navigator as Navigator & { gpu?: unknown }).gpu !== undefined;

    let handle: GlassboxHandle;
    let backend: 'webgpu' | 'cpu';
    if (tryWebGpu) {
        try {
            handle = (await wasm.Glassbox.fromBlobWebGpu(blob)) as GlassboxHandle;
            backend = 'webgpu';
        } catch (e) {
            if (backendPref === 'webgpu') throw e;
            handle = wasm.Glassbox.fromBlob(blob) as GlassboxHandle;
            backend = 'cpu';
        }
    } else {
        handle = wasm.Glassbox.fromBlob(blob) as GlassboxHandle;
        backend = 'cpu';
    }

    onProgress?.({ phase: 'ready', bytes_loaded: received, bytes_total: total });
    return { handle, backend };
}

export function formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function formatParams(n: number): string {
    if (n < 1_000_000) return `${(n / 1_000).toFixed(0)}K`;
    if (n < 1_000_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    return `${(n / 1_000_000_000).toFixed(2)}B`;
}
