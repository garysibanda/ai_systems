use anyhow::Result;
use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Json, Response};
use axum::routing::{get, post};
use axum::Router;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{debug, error, info};

use crate::engine::{EngineApi, InferenceRequest};
use crate::metrics;

#[derive(Clone)]
struct AppState {
    engine: Arc<dyn EngineApi>,
    api_key: Option<String>,
    dev_mode: bool,
    rate_limiter: Arc<SimpleRateLimiter>,
}

struct SimpleRateLimiter {
    max_per_second: u64,
    state: Mutex<(Instant, u64)>,
}

impl SimpleRateLimiter {
    fn new(max_per_second: u64) -> Self {
        Self {
            max_per_second,
            state: Mutex::new((Instant::now(), 0)),
        }
    }

    fn allow(&self) -> bool {
        let mut state = self.state.lock();
        if state.0.elapsed() >= Duration::from_secs(1) {
            *state = (Instant::now(), 0);
        }

        if state.1 >= self.max_per_second {
            return false;
        }

        state.1 += 1;
        true
    }
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub api_key: Option<String>,
    pub dev_mode: bool,
    pub cors_allowed_origins: Vec<String>,
    pub rate_limit_rps: u64,
}

impl ServerConfig {
    pub fn from_env() -> Result<Self> {
        let dev_mode = std::env::var("DEV_MODE")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);

        let api_key = std::env::var("LLM_API_KEY").ok().filter(|v| !v.is_empty());
        if !dev_mode && api_key.is_none() {
            return Err(anyhow::anyhow!(
                "LLM_API_KEY must be set unless DEV_MODE=true"
            ));
        }

        let cors_allowed_origins = std::env::var("CORS_ALLOWED_ORIGINS")
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|_| {
                vec![
                    "http://localhost:3000".to_string(),
                    "http://localhost:5173".to_string(),
                    "http://localhost:8080".to_string(),
                ]
            });

        let rate_limit_rps = std::env::var("RATE_LIMIT_RPS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(60);

        Ok(Self {
            api_key,
            dev_mode,
            cors_allowed_origins,
            rate_limit_rps,
        })
    }
}

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
    stream: Option<bool>,
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f64 {
    0.0
}

fn default_top_p() -> f64 {
    1.0
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
    engine: Arc<dyn EngineApi>,
    port: u16,
    config: ServerConfig,
}

impl InferenceServer {
    pub fn new(engine: Arc<dyn EngineApi>, port: u16, config: ServerConfig) -> Self {
        Self {
            engine,
            port,
            config,
        }
    }

    pub async fn run(self) -> Result<()> {
        let state = AppState {
            engine: self.engine,
            api_key: self.config.api_key.clone(),
            dev_mode: self.config.dev_mode,
            rate_limiter: Arc::new(SimpleRateLimiter::new(self.config.rate_limit_rps)),
        };

        let app = build_app(state, &self.config)?;

        let addr = format!("0.0.0.0:{}", self.port);
        info!("Server listening on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

fn build_app(state: AppState, config: &ServerConfig) -> Result<Router> {
    let api_routes = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            enforce_rate_limit,
        ));

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_endpoint))
        .merge(api_routes)
        .layer(build_cors_layer(&config.cors_allowed_origins)?)
        .with_state(state);

    Ok(app)
}

fn build_cors_layer(origins: &[String]) -> Result<CorsLayer> {
    let mut allowed = Vec::new();
    for origin in origins {
        allowed.push(
            HeaderValue::from_str(origin)
                .map_err(|_| anyhow::anyhow!("Invalid CORS origin: {}", origin))?,
        );
    }

    Ok(CorsLayer::new()
        .allow_origin(AllowOrigin::list(allowed))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([AUTHORIZATION, CONTENT_TYPE]))
}

async fn require_api_key(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if state.dev_mode {
        return Ok(next.run(request).await);
    }

    let expected_key = state.api_key.as_deref().ok_or(StatusCode::UNAUTHORIZED)?;

    let provided = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(parse_bearer_token)
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if provided != expected_key {
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(next.run(request).await)
}

async fn enforce_rate_limit(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if state.rate_limiter.allow() {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::TOO_MANY_REQUESTS)
    }
}

fn parse_bearer_token(header: &str) -> Option<&str> {
    header
        .strip_prefix("Bearer ")
        .map(str::trim)
        .filter(|v| !v.is_empty())
}

fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy"
    }))
}

async fn metrics_endpoint() -> Result<String, StatusCode> {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    String::from_utf8(buffer).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    debug!("Received chat completion request");

    let start_time = Instant::now();

    let prompt_text = request
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt_tokens = state
        .engine
        .tokenize(&prompt_text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if prompt_tokens.len() > state.engine.max_seq_len() {
        metrics::increment_requests("chat/completions", "error");
        return Err(StatusCode::BAD_REQUEST);
    }

    let (tx, rx) = tokio::sync::oneshot::channel();

    let inference_request = InferenceRequest {
        prompt: prompt_tokens.clone(),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        endpoint: "chat/completions",
        response_tx: tx,
        created_at: start_time,
    };

    state.engine.enqueue(inference_request);

    let generated_tokens = rx.await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let generated_tokens = generated_tokens.map_err(|e| {
        error!("Inference error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let completion_text = state
        .engine
        .detokenize(&generated_tokens)
        .unwrap_or_else(|_| tokens_to_text(&generated_tokens));
    let completion_tokens = generated_tokens.len();

    metrics::increment_requests("chat/completions", "success");

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: current_unix_timestamp(),
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
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    debug!("Received completion request");

    let start_time = Instant::now();

    let prompt_tokens = state
        .engine
        .tokenize(&request.prompt)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if prompt_tokens.len() > state.engine.max_seq_len() {
        metrics::increment_requests("completions", "error");
        return Err(StatusCode::BAD_REQUEST);
    }

    let (tx, rx) = tokio::sync::oneshot::channel();

    let inference_request = InferenceRequest {
        prompt: prompt_tokens.clone(),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        endpoint: "completions",
        response_tx: tx,
        created_at: start_time,
    };

    state.engine.enqueue(inference_request);

    let generated_tokens = rx.await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let generated_tokens = generated_tokens.map_err(|e| {
        error!("Inference error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let completion_text = state
        .engine
        .detokenize(&generated_tokens)
        .unwrap_or_else(|_| tokens_to_text(&generated_tokens));
    let completion_tokens = generated_tokens.len();

    metrics::increment_requests("completions", "success");

    let response = CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: current_unix_timestamp(),
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

fn tokens_to_text(tokens: &[u32]) -> String {
    tokens
        .iter()
        .filter_map(|&t| if t < 256 { Some(t as u8 as char) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        build_app, current_unix_timestamp, parse_bearer_token, AppState, ServerConfig,
        SimpleRateLimiter,
    };
    use crate::engine::{EngineApi, InferenceRequest};
    use anyhow::Result;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::util::ServiceExt;

    struct MockEngine {
        prompt_len: usize,
        max_seq_len: usize,
        completion_text: String,
        completion_tokens: Vec<u32>,
    }

    impl EngineApi for MockEngine {
        fn tokenize(&self, _text: &str) -> Result<Vec<u32>> {
            Ok(vec![1; self.prompt_len])
        }

        fn detokenize(&self, _tokens: &[u32]) -> Result<String> {
            Ok(self.completion_text.clone())
        }

        fn max_seq_len(&self) -> usize {
            self.max_seq_len
        }

        fn enqueue(&self, request: InferenceRequest) {
            let _ = request.response_tx.send(Ok(self.completion_tokens.clone()));
        }
    }

    fn test_config() -> ServerConfig {
        ServerConfig {
            api_key: None,
            dev_mode: true,
            cors_allowed_origins: vec!["http://localhost:3000".to_string()],
            rate_limit_rps: 100,
        }
    }

    fn test_state(prompt_len: usize, max_seq_len: usize) -> AppState {
        AppState {
            engine: Arc::new(MockEngine {
                prompt_len,
                max_seq_len,
                completion_text: "hello".to_string(),
                completion_tokens: vec![1, 2, 3],
            }),
            api_key: None,
            dev_mode: true,
            rate_limiter: Arc::new(SimpleRateLimiter::new(100)),
        }
    }

    #[tokio::test]
    async fn health_endpoint_returns_ok() {
        let app = build_app(test_state(1, 100), &test_config()).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_ok() {
        let app = build_app(test_state(1, 100), &test_config()).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn completions_endpoint_returns_response() {
        let app = build_app(test_state(3, 100), &test_config()).unwrap();
        let payload = r#"{"prompt":"test","max_tokens":8,"temperature":0.0,"top_p":1.0}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["choices"][0]["text"], "hello");
    }

    #[tokio::test]
    async fn completions_endpoint_rejects_over_limit_prompt() {
        let app = build_app(test_state(50, 10), &test_config()).unwrap();
        let payload = r#"{"prompt":"test","max_tokens":8,"temperature":0.0,"top_p":1.0}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn bearer_parser_extracts_token() {
        assert_eq!(parse_bearer_token("Bearer abc123"), Some("abc123"));
        assert_eq!(parse_bearer_token("Basic abc123"), None);
    }

    #[test]
    fn unix_timestamp_is_non_zero() {
        assert!(current_unix_timestamp() > 0);
    }
}
