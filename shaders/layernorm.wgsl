struct Dims {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WG: u32 = 256u;

var<workgroup> shared_sum: array<f32, WG>;
var<workgroup> shared_sumsq: array<f32, WG>;

@compute @workgroup_size(256, 1)
fn layernorm_row(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let cols = dims.cols;
    if (row >= dims.rows) { return; }
    let off = row * cols;
    let tid = lid.x;

    var s: f32 = 0.0;
    var sq: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= cols) { break; }
        let v = input[off + i];
        s = s + v;
        sq = sq + v * v;
        i = i + WG;
    }
    shared_sum[tid] = s;
    shared_sumsq[tid] = sq;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
            shared_sumsq[tid] = shared_sumsq[tid] + shared_sumsq[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let n = f32(cols);
    let mean = shared_sum[0] / n;
    let var_ = shared_sumsq[0] / n - mean * mean;
    let inv = 1.0 / sqrt(var_ + dims.eps);

    var j: u32 = tid;
    loop {
        if (j >= cols) { break; }
        let v = input[off + j];
        output[off + j] = (v - mean) * inv * gamma[j] + beta[j];
        j = j + WG;
    }
}
