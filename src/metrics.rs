use prometheus::{HistogramVec, IntCounterVec, IntGauge, register_histogram_vec, register_int_counter_vec, register_int_gauge};

lazy_static::lazy_static! {
    pub static ref TOKENS_PER_SECOND: HistogramVec = register_histogram_vec!(
        "tokens_per_second",
        "Tokens generated per second",
        &["model"]
    )
    .unwrap();

    pub static ref REQUEST_LATENCY: HistogramVec = register_histogram_vec!(
        "request_latency_seconds",
        "Request latency in seconds",
        &["endpoint", "status"]
    )
    .unwrap();

    pub static ref QUEUE_DEPTH: IntGauge = register_int_gauge!(
        "queue_depth",
        "Number of requests waiting in queue"
    )
    .unwrap();

    pub static ref GPU_UTILIZATION: IntGauge = register_int_gauge!(
        "gpu_utilization_percent",
        "GPU utilization percentage"
    )
    .unwrap();

    pub static ref REQUESTS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "requests_total",
        "Total number of requests",
        &["endpoint", "status"]
    )
    .unwrap();

    pub static ref TOKENS_GENERATED: IntCounterVec = register_int_counter_vec!(
        "tokens_generated_total",
        "Total tokens generated",
        &["model"]
    )
    .unwrap();
}

pub fn init_metrics() {
    // Metrics are registered via lazy_static
    tracing::info!("Metrics initialized");
}

pub fn record_tokens_per_second(model: &str, tokens: f64) {
    TOKENS_PER_SECOND.with_label_values(&[model]).observe(tokens);
}

pub fn record_request_latency(endpoint: &str, status: &str, latency: f64) {
    REQUEST_LATENCY
        .with_label_values(&[endpoint, status])
        .observe(latency);
}

pub fn set_queue_depth(depth: i64) {
    QUEUE_DEPTH.set(depth);
}

#[allow(dead_code)]
pub fn set_gpu_utilization(util: i64) {
    GPU_UTILIZATION.set(util);
}

pub fn increment_requests(endpoint: &str, status: &str) {
    REQUESTS_TOTAL.with_label_values(&[endpoint, status]).inc();
}

pub fn increment_tokens_generated(model: &str, count: i64) {
    TOKENS_GENERATED
        .with_label_values(&[model])
        .inc_by(count as u64);
}