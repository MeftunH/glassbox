import type { GlassboxHandle, LoadProgress, ModelInfo } from '$lib/glassbox';

class SessionStore {
    handle = $state<GlassboxHandle | null>(null);
    info = $state<ModelInfo | null>(null);
    progress = $state<LoadProgress | null>(null);
    error = $state<string | null>(null);
    activeBackend = $state<'webgpu' | 'cpu' | null>(null);

    prompt = $state<string>('When the model attends to the previous token,');
    tokens = $state<number[]>([]);
    generated = $state<number[]>([]);
    generatedText = $state<string>('');
    elapsedMs = $state<number>(0);
    tokensPerSecond = $state<number>(0);
    selectedLayer = $state<number>(0);
    selectedHead = $state<number>(0);
    selectedPosition = $state<number>(0);
    isGenerating = $state<boolean>(false);
    maxNewTokens = $state<number>(32);
    temperature = $state<number>(0.8);
    topK = $state<number>(40);
    topP = $state<number>(0.95);

    setHandle(h: GlassboxHandle) {
        this.handle = h;
        this.info = h.modelInfo();
        this.error = null;
    }

    setError(message: string) {
        this.error = message;
        this.handle = null;
        this.info = null;
    }

    reset() {
        this.tokens = [];
        this.generated = [];
        this.selectedLayer = 0;
        this.selectedHead = 0;
        this.selectedPosition = 0;
    }
}

export const session = new SessionStore();
