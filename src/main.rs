use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tracing::{info, error};
use tracing_subscriber::EnvFilter;

mod engine;
mod server;
mod metrics;

use engine::InferenceEngine;
use server::InferenceServer;

#[derive(Parser, Debug)]
#[command(name = "llm-inference-server")]
#[command(about = "High-performance LLM inference server with continuous batching")]
struct Args {
    /// Path to GGUF model file
    #[arg(long, default_value = "")]
    model: String,

    /// Server port
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Maximum batch size
    #[arg(long, default_value_t = 32)]
    max_batch_size: usize,

    /// Maximum sequence length
    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Enable verbose logging
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    // let filter = if args.verbose {
    //     "llm_inference_server=debug,tower_http=debug"
    // } else {
    //     "llm_inference_server=info"
    // };

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    if args.model.is_empty() {
        error!("--model path is required");
        std::process::exit(1);
    }

    info!("Starting LLM inference server");
    info!("Model: {}", args.model);
    info!("Port: {}", args.port);
    info!("Max batch size: {}", args.max_batch_size);
    info!("Max sequence length: {}", args.max_seq_len);

    // Initialize metrics
    metrics::init_metrics();

    // Initialize inference engine
    let engine = Arc::new(
        InferenceEngine::new(
            &args.model,
            args.max_batch_size,
            args.max_seq_len,
        )
        .await?,
    );

    info!("Model loaded successfully");

    // Start the server
    let server = InferenceServer::new(engine, args.port);
    server.run().await?;

    Ok(())
}

