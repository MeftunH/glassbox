struct Dims {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    causal: u32,
    write_pattern: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> pattern: array<f32>;

const NEG_INF: f32 = -3.4e38;

fn q_off(b: u32, h: u32, i: u32) -> u32 {
    return ((b * dims.heads + h) * dims.seq_q + i) * dims.head_dim;
}

fn k_off(b: u32, h: u32, j: u32) -> u32 {
    return ((b * dims.heads + h) * dims.seq_k + j) * dims.head_dim;
}

fn pat_off(b: u32, h: u32, i: u32) -> u32 {
    return ((b * dims.heads + h) * dims.seq_q + i) * dims.seq_k;
}

@compute @workgroup_size(64, 1, 1)
fn attention(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos_global = gid.x;
    let total_rows = dims.batch * dims.heads * dims.seq_q;
    if (pos_global >= total_rows) { return; }

    let i = pos_global % dims.seq_q;
    let h = (pos_global / dims.seq_q) % dims.heads;
    let b = pos_global / (dims.seq_q * dims.heads);

    let scale = 1.0 / sqrt(f32(dims.head_dim));

    var max_score: f32 = NEG_INF;
    var scratch_off = pat_off(b, h, i);

    for (var j: u32 = 0u; j < dims.seq_k; j = j + 1u) {
        if (dims.causal == 1u && j > i) {
            pattern[scratch_off + j] = NEG_INF;
            continue;
        }
        var s: f32 = 0.0;
        let qb = q_off(b, h, i);
        let kb = k_off(b, h, j);
        for (var d: u32 = 0u; d < dims.head_dim; d = d + 1u) {
            s = s + q[qb + d] * k[kb + d];
        }
        s = s * scale;
        pattern[scratch_off + j] = s;
        if (s > max_score) { max_score = s; }
    }

    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < dims.seq_k; j = j + 1u) {
        let s = pattern[scratch_off + j];
        if (s <= NEG_INF * 0.5) {
            pattern[scratch_off + j] = 0.0;
            continue;
        }
        let e = exp(s - max_score);
        pattern[scratch_off + j] = e;
        sum = sum + e;
    }
    if (sum > 0.0) {
        let inv = 1.0 / sum;
        for (var j: u32 = 0u; j < dims.seq_k; j = j + 1u) {
            pattern[scratch_off + j] = pattern[scratch_off + j] * inv;
        }
    }

    let ob = q_off(b, h, i);
    for (var d: u32 = 0u; d < dims.head_dim; d = d + 1u) {
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < dims.seq_k; j = j + 1u) {
            let vb = k_off(b, h, j);
            acc = acc + pattern[scratch_off + j] * v[vb + d];
        }
        out[ob + d] = acc;
    }

    if (dims.write_pattern == 0u) {
        for (var j: u32 = 0u; j < dims.seq_k; j = j + 1u) {
            pattern[scratch_off + j] = 0.0;
        }
    }
}
