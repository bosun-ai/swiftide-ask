mod config;
use std::path::PathBuf;

use anyhow::{Context, Result};
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use swiftide;
// use swiftide::{loaders, node_caches, storage, transformers, IngestionPipeline};

const EMBEDDING_SIZE: usize = 1536;

#[tokio::main]
async fn main() {
    let config = config::Config::from_env();

    // path is the current working directory
    let path = std::env::current_dir().unwrap();

    // namespace is "swiftide-ask-" + the current working directory
    let namespace = "swiftide-ass-".to_string() + path.to_str().unwrap();

    let llm_client = swiftide::integrations::openai::OpenAI::builder()
        .build()
        .expect("Could not build OpenAI client");

    let qdrant_client = QdrantClientConfig::from_url(config.qdrant_url.as_str())
        .with_api_key(config.qdrant_api_key.clone())
        .build()
        .expect("Could not build Qdrant client");

    load_documentation(config, &llm_client, qdrant_client, &namespace, path)
        .await
        .expect("Could not load documentation");
}

async fn load_documentation(
    config: &config::Config,
    llm_client: &swiftide::integrations::openai::OpenAI,
    qdrant_client: QdrantClient,
    namespace: &str,
    path: impl Into<PathBuf>,
) -> Result<()> {
    // Load any documentation
    swiftide::ingestion::IngestionPipeline::from_loader(
        swiftide::loaders::FileLoader::new(path).with_extensions(&["md"]),
    )
    .with_concurrency(50)
    .filter_cached(swiftide::integrations::redis::RedisNodeCache::try_from_url(
        config.redis_url.as_deref().context("Expected redis url")?,
        namespace,
    )?)
    .then_chunk(swiftide::transformers::ChunkMarkdown::with_chunk_range(
        100..5000,
    ))
    .then(swiftide::transformers::MetadataQAText::new(
        llm_client.clone(),
    ))
    .then_in_batch(
        100,
        swiftide::transformers::OpenAIEmbed::new(llm_client.clone()),
    )
    .store_with(
        swiftide::integrations::qdrant::Qdrant::builder()
            .client(qdrant_client)
            .batch_size(Some(1024))
            .vector_size(EMBEDDING_SIZE)
            .build()?,
    )
    .run()
    .await?;

    Ok(())
}
