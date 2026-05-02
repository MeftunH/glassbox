use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::error::{ModelError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeBlob {
    pub vocab: AHashMap<String, u32>,
    pub merges: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct Bpe {
    vocab: AHashMap<String, u32>,
    inv_vocab: Vec<String>,
    merge_rank: AHashMap<(String, String), u32>,
}

impl Bpe {
    pub fn from_blob(blob: BpeBlob) -> Self {
        let mut inv = vec![String::new(); blob.vocab.len()];
        for (token, &id) in &blob.vocab {
            if (id as usize) < inv.len() {
                inv[id as usize] = token.clone();
            }
        }
        let merge_rank: AHashMap<_, _> = blob
            .merges
            .into_iter()
            .enumerate()
            .map(|(rank, pair)| (pair, rank as u32))
            .collect();
        Self { vocab: blob.vocab, inv_vocab: inv, merge_rank }
    }

    pub fn from_json(s: &str) -> Result<Self> {
        let blob: BpeBlob = serde_json::from_str(s).map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        Ok(Self::from_blob(blob))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.inv_vocab.get(id as usize).map(String::as_str)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let pre = pretokenise(text);
        let mut out = Vec::with_capacity(pre.len());
        for word in pre {
            for piece in self.bpe(&word) {
                if let Some(id) = self.vocab.get(&piece) {
                    out.push(*id);
                }
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            if let Some(t) = self.id_to_token(id) {
                s.push_str(&unbyte(t));
            }
        }
        s
    }

    fn bpe(&self, word: &str) -> Vec<String> {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if symbols.len() < 2 {
            return symbols;
        }
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    if best.map(|(_, r)| rank < r).unwrap_or(true) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..=i + 1, std::iter::once(merged));
            if symbols.len() < 2 {
                break;
            }
        }
        symbols
    }
}

fn pretokenise(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for c in text.chars() {
        if c.is_whitespace() && !buf.is_empty() {
            out.push(std::mem::take(&mut buf));
        }
        buf.push(c);
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

fn unbyte(s: &str) -> String {
    s.replace('\u{0120}', " ").replace('\u{010A}', "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vocab_does_not_panic() {
        let bpe = Bpe::from_blob(BpeBlob { vocab: AHashMap::new(), merges: vec![] });
        assert_eq!(bpe.vocab_size(), 0);
        assert_eq!(bpe.encode("hello"), Vec::<u32>::new());
    }
}
