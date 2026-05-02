struct Dims {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const WG: u32 = 256u;

var<workgroup> shared_max: array<f32, WG>;
var<workgroup> shared_sum: array<f32, WG>;

@compute @workgroup_size(256, 1)
fn softmax_row(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let cols = dims.cols;
    if (row >= dims.rows) {
        return;
    }
    let row_off = row * cols;
    let tid = lid.x;

    var local_max: f32 = -3.4e38;
    var i: u32 = tid;
    loop {
        if (i >= cols) { break; }
        local_max = max(local_max, input[row_off + i]);
        i = i + WG;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = shared_max[0];

    var local_sum: f32 = 0.0;
    var j: u32 = tid;
    loop {
        if (j >= cols) { break; }
        let e = exp(input[row_off + j] - row_max);
        output[row_off + j] = e;
        local_sum = local_sum + e;
        j = j + WG;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = shared_sum[0];

    if (row_sum > 0.0) {
        let inv = 1.0 / row_sum;
        var k: u32 = tid;
        loop {
            if (k >= cols) { break; }
            output[row_off + k] = output[row_off + k] * inv;
            k = k + WG;
        }
    }
}
