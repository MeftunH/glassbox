use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: Some(40), top_p: Some(0.95), seed: 0 }
    }
}

#[derive(Debug)]
pub struct Sampler {
    rng: StdRng,
    cfg: SamplingConfig,
}

impl Sampler {
    pub fn new(cfg: SamplingConfig) -> Self {
        Self { rng: StdRng::seed_from_u64(cfg.seed), cfg }
    }

    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best_idx = 0u32;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                best_idx = i as u32;
            }
        }
        best_idx
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.cfg.temperature <= f32::EPSILON {
            return Self::argmax(logits);
        }

        let mut scaled: Vec<f32> = logits.iter().map(|&x| x / self.cfg.temperature).collect();

        if let Some(k) = self.cfg.top_k {
            top_k_mask(&mut scaled, k);
        }

        let mut probs = softmax(&scaled);

        if let Some(p) = self.cfg.top_p {
            top_p_mask(&mut probs, p);
            renormalise(&mut probs);
        }

        let mut r: f32 = self.rng.gen();
        for (i, &prob) in probs.iter().enumerate() {
            r -= prob;
            if r <= 0.0 {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for &x in xs {
        if x > max {
            max = x;
        }
    }
    let mut sum = 0.0f32;
    let mut out: Vec<f32> = xs
        .iter()
        .map(|&x| {
            let e = (x - max).exp();
            sum += e;
            e
        })
        .collect();
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

fn top_k_mask(xs: &mut [f32], k: usize) {
    if k == 0 || k >= xs.len() {
        return;
    }
    let mut sorted: Vec<f32> = xs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    for v in xs.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn top_p_mask(probs: &mut [f32], p: f32) {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0f32;
    let mut keep = vec![false; probs.len()];
    for (idx, prob) in indexed {
        cumulative += prob;
        keep[idx] = true;
        if cumulative >= p {
            break;
        }
    }
    for (i, k) in keep.iter().enumerate() {
        if !k {
            probs[i] = 0.0;
        }
    }
}

fn renormalise(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(Sampler::argmax(&[0.1, 0.5, 0.3, 0.2]), 1);
    }

    #[test]
    fn temperature_zero_is_argmax() {
        let cfg = SamplingConfig { temperature: 0.0, top_k: None, top_p: None, seed: 7 };
        let mut s = Sampler::new(cfg);
        assert_eq!(s.sample(&[2.0, 9.0, 1.0]), 1);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }
}
