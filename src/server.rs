use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info};

use crate::engine::{InferenceEngine, InferenceRequest};
use crate::metrics;

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
    #[allow(dead_code)]
    stream: Option<bool>,  // Reserved for streaming support
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f64 {
    0.7
}

fn default_top_p() -> f64 {
    0.9
}

#[derive(Serialize)]
pub struct ChatCompletionChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
pub struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: String,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

pub struct InferenceServer {
    engine: Arc<InferenceEngine>,
    port: u16,
}

impl InferenceServer {
    pub fn new(engine: Arc<InferenceEngine>, port: u16) -> Self {
        Self { engine, port }
    }

    pub async fn run(self) -> Result<()> {
        // Start batch processor
        self.engine.clone().start_batch_processor();

        let app = Router::new()
            .route("/health", get(health_check))
            .route("/metrics", get(metrics_endpoint))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .layer(CorsLayer::permissive())
            .with_state(self.engine.clone());

        let addr = format!("0.0.0.0:{}", self.port);
        info!("Server listening on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy"
    }))
}

async fn metrics_endpoint() -> String {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

async fn chat_completions(
    State(engine): State<Arc<InferenceEngine>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    debug!("Received chat completion request");

    let start_time = Instant::now();

    // Convert messages to prompt tokens
    let prompt_text = request
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Tokenize using the engine's tokenizer
    let prompt_tokens = engine.tokenize(&prompt_text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let prompt_len = prompt_tokens.len();

    if prompt_len > engine.max_seq_len() {
        metrics::increment_requests("chat/completions", "error");
        return Err(StatusCode::BAD_REQUEST);
    }

    let (tx, rx) = tokio::sync::oneshot::channel();

    let inference_request = InferenceRequest {
        prompt: prompt_tokens.clone(),  // Clone to keep a copy for usage stats
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        response_tx: tx,
        created_at: start_time,
    };

    engine.enqueue(inference_request);

    let generated_tokens = rx.await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let generated_tokens = generated_tokens.map_err(|e| {
        error!("Inference error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let completion_text = engine.detokenize(&generated_tokens)
        .unwrap_or_else(|_| tokens_to_text(&generated_tokens));
    let completion_tokens = generated_tokens.len();

    metrics::increment_requests("chat/completions", "success");

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: start_time.elapsed().as_secs(),
        model: request.model.unwrap_or_else(|| "default".to_string()),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: completion_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: prompt_tokens.len() + completion_tokens,
        },
    };

    Ok(Json(response))
}

async fn completions(
    State(engine): State<Arc<InferenceEngine>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    debug!("Received completion request");

    let start_time = Instant::now();

    let prompt_tokens = engine.tokenize(&request.prompt)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let prompt_len = prompt_tokens.len();

    if prompt_len > engine.max_seq_len() {
        metrics::increment_requests("completions", "error");
        return Err(StatusCode::BAD_REQUEST);
    }

    let (tx, rx) = tokio::sync::oneshot::channel();

    let inference_request = InferenceRequest {
        prompt: prompt_tokens.clone(),  // Clone to keep a copy for usage stats
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        response_tx: tx,
        created_at: start_time,
    };

    engine.enqueue(inference_request);

    let generated_tokens = rx.await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let generated_tokens = generated_tokens.map_err(|e| {
        error!("Inference error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let completion_text = engine.detokenize(&generated_tokens)
        .unwrap_or_else(|_| tokens_to_text(&generated_tokens));
    let completion_tokens = generated_tokens.len();

    metrics::increment_requests("completions", "success");

    let response = CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: start_time.elapsed().as_secs(),
        model: request.model.unwrap_or_else(|| "default".to_string()),
        choices: vec![CompletionChoice {
            text: completion_text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: prompt_tokens.len() + completion_tokens,
        },
    };

    Ok(Json(response))
}

// Fallback detokenization if engine detokenizer fails
fn tokens_to_text(tokens: &[u32]) -> String {
    tokens
        .iter()
        .filter_map(|&t| {
            if t < 256 {
                Some(t as u8 as char)
            } else {
                None
            }
        })
        .collect()
}