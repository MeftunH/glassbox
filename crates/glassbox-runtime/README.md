# glassbox-runtime

Op dispatch and backends. Defines the `Backend` trait and provides two implementations: `CpuBackend` (rayon-parallel f32, the reference) and `WgpuBackend` (WGSL kernels).
