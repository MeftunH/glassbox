use std::env;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use glassbox_models::{Bpe, GlxFile, Gpt2, Gpt2Runner};
use glassbox_runtime::{CpuBackend, HookRegistry, Sampler, SamplingConfig};

fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| "models/gpt2-small.glx".into());
    let prompt = env::args().nth(2).unwrap_or_else(|| "The transformer learns to attend to".into());
    let max_new: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    eprintln!("loading {path} ...");
    let t0 = Instant::now();
    let reader = BufReader::new(File::open(&path)?);
    let glx = GlxFile::read(reader)?;
    let model = Gpt2::from_glx(&glx)?;
    let tokenizer_blob = glx.header.tokenizer_blob.clone().unwrap_or_default();
    let tokenizer = Bpe::from_json(&tokenizer_blob)?;
    eprintln!(
        "  ok: {n} tensors, {p}M params, loaded in {ms} ms",
        n = glx.header.tensors.len(),
        p = model.parameter_count() / 1_000_000,
        ms = t0.elapsed().as_millis()
    );

    let backend = CpuBackend;
    let hooks = HookRegistry::new();
    let runner = Gpt2Runner::new(&model, &backend, hooks)?;
    let mut sampler = Sampler::new(SamplingConfig { temperature: 0.0, top_k: None, top_p: None, seed: 0 });

    let mut ids: Vec<u32> = tokenizer.encode(&prompt);
    eprintln!("prompt tokens: {ids:?}");

    print!("{prompt}");
    let t_gen = Instant::now();
    for _ in 0..max_new {
        let logits = runner.forward(&ids)?;
        let last = runner.last_position_logits(&logits)?;
        let next = sampler.sample(&last);
        ids.push(next);
        print!("{}", tokenizer.decode(&[next]));
    }
    println!();
    eprintln!(
        "{n} tokens in {ms} ms ({tps:.1} tok/s)",
        n = max_new,
        ms = t_gen.elapsed().as_millis(),
        tps = max_new as f64 / t_gen.elapsed().as_secs_f64()
    );
    Ok(())
}
