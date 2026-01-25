use anyhow::Result;
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_transformers::models::phi3::Config;
use candle_transformers::models::quantized_phi3::ModelWeights as QuantizedPhi3;
use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use std::time::{Instant, Duration};
use tracing::{debug, warn};
use tokenizers::Tokenizer;

use crate::metrics;

/// Paged KV cache entry for a single sequence
/// (Reserved for future paged attention implementation)
#[derive(Clone)]
#[allow(dead_code)]
struct KVPage {
    k_cache: Tensor,
    v_cache: Tensor,
    seq_len: usize,
}

/// Request in the batch queue
pub struct InferenceRequest {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub response_tx: tokio::sync::oneshot::Sender<Result<Vec<u32>>>,
    pub created_at: Instant,
}

/// Continuous batching inference engine
pub struct InferenceEngine {
    model: Arc<RwLock<QuantizedPhi3>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    max_batch_size: usize,
    max_seq_len: usize,
    #[allow(dead_code)]
    config: Config,  // Reserved for future use
    #[allow(dead_code)]
    kv_cache: Arc<DashMap<usize, KVPage>>,  // Reserved for paged attention
    request_queue: Arc<RwLock<Vec<InferenceRequest>>>,
    next_request_id: Arc<RwLock<usize>>,
}

impl InferenceEngine {
    pub async fn new(
        model_path: &str,
        max_batch_size: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        // Use Metal device for Apple Silicon
        let device = Device::new_metal(0)
            .unwrap_or_else(|_| {
                warn!("Metal device not available, falling back to CPU");
                Device::Cpu
            });

        debug!("Loading model from: {}", model_path);

        // Load GGUF model using Candle's quantized GGUF API
        let model_path_buf = std::path::PathBuf::from(model_path);
        
        // Load quantized GGUF file
        let mut file = std::fs::File::open(&model_path_buf)?;
        
        // Read GGUF content
        let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        
        // Create model from GGUF (use_flash_attn: bool, content, device)
        let use_flash_attn = false;  // Set to true if you want flash attention
        let model = QuantizedPhi3::from_gguf(use_flash_attn, gguf_content, &mut file, &device)?;

        // For quantized models, create a config for compatibility
        let config = Config {
            hidden_size: 3072,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            intermediate_size: 8192,
            max_position_embeddings: 4096,
            original_max_position_embeddings: Some(4096),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            vocab_size: 32064,
            hidden_act: candle_nn::Activation::Gelu,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            partial_rotary_factor: Some(0.5),
            tie_word_embeddings: false,
        };

        // Load tokenizer - try to find tokenizer.json in same directory
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
            return Err(anyhow::anyhow!("Tokenizer file not found at {:?}", tokenizer_path));
        };

        debug!("Model loaded successfully");

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            max_batch_size,
            max_seq_len,
            config,
            kv_cache: Arc::new(DashMap::new()),
            request_queue: Arc::new(RwLock::new(Vec::new())),
            next_request_id: Arc::new(RwLock::new(0)),
        })
    }

    /// Tokenize text using the model's tokenizer
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {:?}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Convert tokens back to text
    pub fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Detokenization error: {:?}", e))
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Add a request to the queue
    pub fn enqueue(&self, request: InferenceRequest) {
        let mut queue = self.request_queue.write();
        queue.push(request);
        metrics::set_queue_depth(queue.len() as i64);
    }

    /// Process the batch queue and generate tokens
    pub async fn process_batch(&self) -> Result<()> {
        let batch: Vec<InferenceRequest> = {
            let mut queue = self.request_queue.write();
            if queue.is_empty() {
                return Ok(());
            }

            // Take up to max_batch_size requests
            let batch_size = queue.len().min(self.max_batch_size);
            queue.drain(..batch_size).collect()
        };

        metrics::set_queue_depth(0);

        if batch.is_empty() {
            return Ok(());
        }

        debug!("Processing batch of {} requests", batch.len());

        // Process requests sequentially
        let start_time = Instant::now();
        let mut total_tokens = 0;

        for (_idx, request) in batch.into_iter().enumerate() {
            let request_id = *self.next_request_id.read();
            *self.next_request_id.write() = request_id + 1;

            match self.generate_tokens(&request.prompt, request.max_tokens, request.temperature, request.top_p).await {
                Ok(tokens) => {
                    total_tokens += tokens.len();
                    let latency = request.created_at.elapsed().as_secs_f64();
                    metrics::record_request_latency("chat/completions", "success", latency);
                    let _ = request.response_tx.send(Ok(tokens));
                }
                Err(e) => {
                    let latency = request.created_at.elapsed().as_secs_f64();
                    metrics::record_request_latency("chat/completions", "error", latency);
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

    /// Generate tokens for a single request
    async fn generate_tokens(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f64,
        _top_p: f64,
    ) -> Result<Vec<u32>> {
        let mut model = self.model.write();
        let mut tokens = prompt.to_vec();
        let mut generated = Vec::new();

        for _ in 0..max_tokens {
            let input_len = tokens.len();
            if input_len >= self.max_seq_len {
                break;
            }

            // Prepare input tensor (add batch dim)
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

            // Forward pass - quantized Phi3 takes input and position offset
            let seqlen_offset = tokens.len().saturating_sub(1);
            let logits = model.forward(&input, seqlen_offset)?;

            // Sample next token
            let next_token = self.sample_token(&logits, temperature)?;
            tokens.push(next_token);
            generated.push(next_token);

            // Check for EOS token
            if next_token == 0 || next_token == 2 {
                break;
            }
        }

        Ok(generated)
    }

    fn sample_token(&self, logits: &Tensor, temperature: f64) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
        
        let logits = if temperature > 0.0 {
            (&last_logits / temperature)?
        } else {
            last_logits
        };

        // Apply softmax
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;
        
        // Simple argmax sampling
        let mut max_idx = 0;
        let mut max_val = probs_vec[0];
        for (idx, &val) in probs_vec.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        Ok(max_idx as u32)
    }

    fn _hash_prompt(&self, prompt: &[u32]) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        hasher.finish() as usize
    }

    /// Start the batch processing loop
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