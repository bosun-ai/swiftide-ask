mod config;
use std::path::PathBuf;

use anyhow::Result;

use swiftide::{
    indexing::{
        self,
        loaders::FileLoader,
        transformers::{
            ChunkCode, ChunkMarkdown, Embed, MetadataQACode, MetadataQAText, OutlineCodeTreeSitter,
        },
    },
    integrations::{self, fastembed::FastEmbed, qdrant::Qdrant, redis::Redis},
    query::{self, answers, query_transformers, response_transformers},
    traits::SimplePrompt,
};

use opentelemetry::trace::TracerProvider as _;
use tracing::instrument;
use tracing_opentelemetry;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

const EMBEDDING_SIZE: u64 = 384; // fastembed vector size
const SUPPORTED_CODE_EXTENSIONS: [&str; 1] = ["rs"];

#[instrument]
#[tokio::main]
async fn main() {
    let fmt_layer = tracing_subscriber::fmt::layer();

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(opentelemetry_otlp::new_exporter().tonic())
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .expect("Couldn't create OTLP tracer")
        .tracer("swiftide-ask");

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    let config = config::Config::from_env();

    let path = std::env::current_dir().unwrap();

    let namespace = format!(
        "swiftide-ask-v0.18-{}-groq",
        path.to_string_lossy().replace("/", "-")
    );

    let llm_client = integrations::groq::Groq::default()
        .with_default_prompt_model("llama-3.1-8b-instant")
        .to_owned();

    // let llm_client = integrations::ollama::Ollama::default()
    //     .with_default_prompt_model("llama3.1")
    //     .to_owned();

    // let llm_client = integrations::openai::OpenAI::builder()
    //     .default_embed_model("text-embedding-3-small")
    //     .default_prompt_model("gpt-4o-mini")
    //     .build()
    //     .expect("Could not create OpenAI");

    let fastembed =
        integrations::fastembed::FastEmbed::try_default().expect("Could not create FastEmbed");

    let qdrant = Qdrant::builder()
        .batch_size(50)
        .vector_size(EMBEDDING_SIZE)
        .collection_name(namespace.as_str())
        .build()
        .expect("Could not create Qdrant");

    let redis_url = config.redis_url.as_deref().expect("Expected redis url");

    let redis = Redis::try_from_url(redis_url, namespace).expect("Could not create Redis");

    load_codebase(
        llm_client.clone(),
        fastembed.clone(),
        qdrant.clone(),
        redis.clone(),
        path.clone(),
    )
    .await
    .expect("Could not load documentation");

    let question = std::env::args()
        .nth(1)
        .expect("Expected question as argument");

    let answer = ask(
        llm_client.clone(),
        fastembed.clone(),
        qdrant.clone(),
        question,
    )
    .await
    .expect("Could not ask question");

    println!("{}", answer);
}

#[instrument]
async fn ask(
    llm_client: impl SimplePrompt + Clone + 'static,
    embed: FastEmbed,
    qdrant: Qdrant,
    question: String,
) -> Result<String> {
    let pipeline = query::Pipeline::default()
        .then_transform_query(query_transformers::GenerateSubquestions::from_client(
            llm_client.clone(),
        ))
        .then_transform_query(query_transformers::Embed::from_client(embed))
        .then_retrieve(qdrant.clone())
        .then_transform_response(response_transformers::Summary::from_client(
            llm_client.clone(),
        ))
        .then_answer(answers::Simple::from_client(llm_client.clone()));

    let result = pipeline.query(question).await?;

    Ok(result.answer().into())
}

#[instrument]
async fn load_codebase(
    llm_client: impl SimplePrompt + Clone + 'static,
    embed: FastEmbed,
    qdrant: Qdrant,
    redis: Redis,
    path: PathBuf,
) -> Result<()> {
    indexing::Pipeline::from_loader(FileLoader::new(path.clone()).with_extensions(&["md"]))
        .filter_cached(redis.clone())
        .then_chunk(ChunkMarkdown::from_chunk_range(100..5000))
        .then(MetadataQAText::new(llm_client.clone()))
        .then_in_batch(100, Embed::new(embed))
        .then_store_with(qdrant.clone())
        .run()
        .await?;

    let code_chunk_size = 2048;

    indexing::Pipeline::from_loader(
        FileLoader::new(path.clone()).with_extensions(&SUPPORTED_CODE_EXTENSIONS),
    )
    .filter_cached(redis)
    .then(OutlineCodeTreeSitter::try_for_language(
        "rust",
        Some(code_chunk_size),
    )?)
    .then_chunk(ChunkCode::try_for_language_and_chunk_size(
        "rust",
        10..code_chunk_size,
    )?)
    .log_errors()
    .filter_errors()
    .then(MetadataQACode::new(llm_client.clone()))
    .then_in_batch(10, Embed::new(FastEmbed::builder().batch_size(10).build()?))
    .then_store_with(qdrant.clone())
    .run()
    .await
}
