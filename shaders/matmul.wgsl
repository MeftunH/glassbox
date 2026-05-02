struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TILE: u32 = 16u;

var<workgroup> a_tile: array<array<f32, TILE>, TILE>;
var<workgroup> b_tile: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(16, 16)
fn matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let m = dims.m;
    let k = dims.k;
    let n = dims.n;

    var acc: f32 = 0.0;
    let tile_count = (k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < tile_count; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        let b_row = t * TILE + lid.y;

        if (row < m && a_col < k) {
            a_tile[lid.y][lid.x] = a[row * k + a_col];
        } else {
            a_tile[lid.y][lid.x] = 0.0;
        }

        if (b_row < k && col < n) {
            b_tile[lid.y][lid.x] = b[b_row * n + col];
        } else {
            b_tile[lid.y][lid.x] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            acc = acc + a_tile[lid.y][i] * b_tile[i][lid.x];
        }

        workgroupBarrier();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}
