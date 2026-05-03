struct Dims {
    n_ids: u32,
    dim: u32,
    vocab: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x;
    if (pos >= dims.n_ids) { return; }
    let id = ids[pos];
    if (id >= dims.vocab) { return; }
    let src_off = id * dims.dim;
    let dst_off = pos * dims.dim;
    for (var d: u32 = 0u; d < dims.dim; d = d + 1u) {
        out[dst_off + d] = table[src_off + d];
    }
}
