use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::quantized_phi3::ModelWeights as QuantizedPhi3;
use parking_lot::RwLock;
use rand::distributions::{Distribution, WeightedIndex};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tracing::{debug, warn};

use crate::metrics;

pub struct InferenceRequest {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub endpoint: &'static str,
    pub response_tx: tokio::sync::oneshot::Sender<Result<Vec<u32>>>,
    pub created_at: Instant,
}

pub trait EngineApi: Send + Sync {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    fn max_seq_len(&self) -> usize;
    fn enqueue(&self, request: InferenceRequest);
}

pub struct InferenceEngine {
    model: Arc<RwLock<QuantizedPhi3>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    max_batch_size: usize,
    max_seq_len: usize,
    request_queue: Arc<RwLock<Vec<InferenceRequest>>>,
}

impl EngineApi for InferenceEngine {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenize(text)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.detokenize(tokens)
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len()
    }

    fn enqueue(&self, request: InferenceRequest) {
        self.enqueue(request)
    }
}

impl InferenceEngine {
    pub async fn new(model_path: &str, max_batch_size: usize, max_seq_len: usize) -> Result<Self> {
        let device = Device::new_metal(0).unwrap_or_else(|_| {
            warn!("Metal device not available, falling back to CPU");
            Device::Cpu
        });

        debug!("Loading model from: {}", model_path);

        let model_path_buf = std::path::PathBuf::from(model_path);
        let mut file = std::fs::File::open(&model_path_buf)?;
        let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

        let use_flash_attn = false;
        let model = QuantizedPhi3::from_gguf(use_flash_attn, gguf_content, &mut file, &device)?;

        let tokenizer_path = model_path_buf
            .parent()
            .map(|p| p.join("tokenizer.json"))
            .unwrap_or_else(|| {
                let mut path = model_path_buf.clone();
                path.set_extension("json");
                path
            });

        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {:?}", e))?
        } else {
            return Err(anyhow::anyhow!(
                "Tokenizer file not found at {:?}",
                tokenizer_path
            ));
        };

        debug!("Model loaded successfully");

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            max_batch_size,
            max_seq_len,
            request_queue: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {:?}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Detokenization error: {:?}", e))
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn enqueue(&self, request: InferenceRequest) {
        let mut queue = self.request_queue.write();
        queue.push(request);
        metrics::set_queue_depth(queue.len() as i64);
    }

    pub async fn process_batch(&self) -> Result<()> {
        let (batch, remaining_depth): (Vec<InferenceRequest>, usize) = {
            let mut queue = self.request_queue.write();
            if queue.is_empty() {
                return Ok(());
            }

            let batch_size = queue.len().min(self.max_batch_size);
            let batch = queue.drain(..batch_size).collect();
            (batch, queue.len())
        };

        metrics::set_queue_depth(remaining_depth as i64);

        if batch.is_empty() {
            return Ok(());
        }

        debug!("Processing batch of {} requests", batch.len());

        let start_time = Instant::now();
        let mut total_tokens = 0;

        for request in batch {
            match self
                .generate_tokens(
                    &request.prompt,
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                )
                .await
            {
                Ok(tokens) => {
                    total_tokens += tokens.len();
                    let latency = request.created_at.elapsed().as_secs_f64();
                    metrics::record_request_latency(request.endpoint, "success", latency);
                    let _ = request.response_tx.send(Ok(tokens));
                }
                Err(e) => {
                    let latency = request.created_at.elapsed().as_secs_f64();
                    metrics::record_request_latency(request.endpoint, "error", latency);
                    let _ = request.response_tx.send(Err(e));
                }
            }
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let tps = total_tokens as f64 / elapsed;
            metrics::record_tokens_per_second("default", tps);
            metrics::increment_tokens_generated("default", total_tokens as i64);
        }

        Ok(())
    }

    async fn generate_tokens(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<Vec<u32>> {
        let mut model = self.model.write();
        let mut tokens = prompt.to_vec();
        let mut generated = Vec::new();

        debug!("Starting generation with {} prompt tokens", tokens.len());

        for i in 0..max_tokens {
            if tokens.len() >= self.max_seq_len {
                debug!("Reached max sequence length");
                break;
            }

            let input_tokens = if i == 0 {
                &tokens[..]
            } else {
                &tokens[tokens.len() - 1..]
            };

            let input = Tensor::new(input_tokens, &self.device)?.unsqueeze(0)?;
            let seqlen_offset = if i == 0 { 0 } else { tokens.len() - 1 };

            debug!(
                "Forward pass iteration {}, seqlen_offset: {}, input_tokens: {}",
                i,
                seqlen_offset,
                input_tokens.len()
            );

            let logits = model.forward(&input, seqlen_offset)?;
            let next_token = self.sample_token(&logits, temperature, top_p)?;
            debug!("Generated token: {}", next_token);

            tokens.push(next_token);
            generated.push(next_token);

            if next_token == 0 || next_token == 2 || next_token == 32000 {
                debug!("EOS token detected: {}", next_token);
                break;
            }
        }

        debug!("Generation complete: {} tokens generated", generated.len());
        Ok(generated)
    }

    fn sample_token(&self, logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;

        let last_logits = match logits.dims().len() {
            3 => {
                let seq_len = logits.dim(1)?;
                logits.i((.., seq_len - 1, ..))?
            }
            2 | 1 => logits.clone(),
            _ => {
                return Err(anyhow::anyhow!(
                    "Unexpected logits shape: {:?}",
                    logits.dims()
                ));
            }
        };

        let last_logits = if last_logits.dims().len() > 1 {
            last_logits.squeeze(0)?
        } else {
            last_logits
        };

        if last_logits.dims().len() != 1 {
            return Err(anyhow::anyhow!(
                "Expected 1D logits after squeeze, got {:?}",
                last_logits.dims()
            ));
        }

        let scaled_logits = if temperature > 0.0 && temperature != 1.0 {
            (last_logits / temperature)?
        } else {
            last_logits
        };

        let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        if probs_vec.is_empty() {
            return Err(anyhow::anyhow!("Empty probability distribution"));
        }

        if temperature <= 0.0 {
            let greedy = argmax_index(&probs_vec)?;
            debug!("Sampled token via greedy decoding: {}", greedy);
            return Ok(greedy as u32);
        }

        let filtered = apply_top_p(&probs_vec, top_p);
        let sampled = sample_weighted(&filtered)?;
        debug!("Sampled token via stochastic decoding: {}", sampled);
        Ok(sampled as u32)
    }

    pub fn start_batch_processor(self: Arc<Self>) {
        let engine = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(10));
            loop {
                interval.tick().await;
                if let Err(e) = engine.process_batch().await {
                    warn!("Batch processing error: {}", e);
                }
            }
        });
    }
}

fn argmax_index(values: &[f32]) -> Result<usize> {
    if values.is_empty() {
        return Err(anyhow::anyhow!("Cannot compute argmax of empty slice"));
    }

    let mut max_idx = 0;
    let mut max_val = values[0];
    for (idx, &val) in values.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    Ok(max_idx)
}

fn apply_top_p(probs: &[f32], top_p: f64) -> Vec<(usize, f32)> {
    if probs.is_empty() {
        return Vec::new();
    }

    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| *p > 0.0)
        .collect();

    if indexed.is_empty() {
        let idx = argmax_index(probs).unwrap_or(0);
        return vec![(idx, 1.0)];
    }

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if !(0.0..1.0).contains(&top_p) {
        let norm: f32 = indexed.iter().map(|(_, p)| *p).sum();
        return indexed
            .into_iter()
            .map(|(idx, p)| (idx, p / norm))
            .collect();
    }

    let mut kept = Vec::new();
    let mut cumulative = 0.0;
    for (idx, p) in indexed {
        kept.push((idx, p));
        cumulative += p as f64;
        if cumulative >= top_p {
            break;
        }
    }

    let norm: f32 = kept.iter().map(|(_, p)| *p).sum();
    if norm <= f32::EPSILON {
        return vec![(kept[0].0, 1.0)];
    }

    kept.into_iter().map(|(idx, p)| (idx, p / norm)).collect()
}

fn sample_weighted(candidates: &[(usize, f32)]) -> Result<usize> {
    if candidates.is_empty() {
        return Err(anyhow::anyhow!("No candidates available for sampling"));
    }

    if candidates.len() == 1 {
        return Ok(candidates[0].0);
    }

    let weights: Vec<f32> = candidates.iter().map(|(_, p)| *p).collect();
    let dist = WeightedIndex::new(weights)
        .map_err(|e| anyhow::anyhow!("Invalid sampling distribution: {:?}", e))?;
    let mut rng = rand::thread_rng();
    let sampled_idx = dist.sample(&mut rng);
    Ok(candidates[sampled_idx].0)
}

#[cfg(test)]
mod tests {
    use super::{apply_top_p, argmax_index};

    #[test]
    fn argmax_returns_index_of_largest_value() {
        let values = vec![0.1, 0.7, 0.2];
        assert_eq!(argmax_index(&values).unwrap(), 1);
    }

    #[test]
    fn top_p_keeps_highest_mass() {
        let probs = vec![0.6, 0.2, 0.1, 0.1];
        let filtered = apply_top_p(&probs, 0.7);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, 0);
        assert_eq!(filtered[1].0, 1);
    }

    #[test]
    fn top_p_invalid_value_keeps_distribution() {
        let probs = vec![0.5, 0.3, 0.2];
        let filtered = apply_top_p(&probs, 1.5);
        assert_eq!(filtered.len(), 3);
        let total: f32 = filtered.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
