mod config;
use std::path::PathBuf;

use anyhow::Result;
use std::io::Write;

use swiftide::{
    indexing::{
        self,
        loaders::FileLoader,
        transformers::{
            ChunkCode, ChunkMarkdown, CompressCodeOutline, Embed, MetadataQACode, MetadataQAText,
            OutlineCodeTreeSitter,
        },
    },
    integrations::{self, fastembed::FastEmbed, ollama::Ollama, qdrant::Qdrant, redis::Redis},
    query::{self, answers, query_transformers, response_transformers},
};

const EMBEDDING_SIZE: u64 = 3072;
const SUPPORTED_CODE_EXTENSIONS: [&str; 1] = ["rs"];

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let config = config::Config::from_env();

    let path = std::env::current_dir().unwrap();

    let namespace = format!(
        "swiftide-ask-v0.11-{}",
        path.to_string_lossy().replace("/", "-")
    );

    let llm_client = integrations::ollama::Ollama::default()
        .with_default_prompt_model("llama3.1")
        .to_owned();

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
        &llm_client,
        fastembed.clone(),
        qdrant.clone(),
        redis.clone(),
        path.clone(),
    )
    .await
    .expect("Could not load documentation");

    loop {
        println!("Ask a question about your code");
        let mut question = String::new();
        print!("\n(q to quit) > ");
        std::io::stdout().flush().unwrap();
        std::io::stdin()
            .read_line(&mut question)
            .expect("Failed to read line");

        question = question.trim().to_string();
        if &question == "q" {
            break;
        }

        let answer = ask(&llm_client, fastembed.clone(), qdrant.clone(), question)
            .await
            .expect("Could not ask question");
        println!("{}", answer);
    }
}

async fn ask(
    llm_client: &Ollama,
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

async fn load_codebase(
    llm_client: &Ollama,
    embed: FastEmbed,
    qdrant: Qdrant,
    redis: Redis,
    path: PathBuf,
) -> Result<()> {
    // Load any documentation
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
    .then(CompressCodeOutline::new(llm_client.clone()))
    .log_errors()
    .filter_errors()
    .then(MetadataQACode::new(llm_client.clone()))
    .then_in_batch(10, Embed::new(FastEmbed::builder().batch_size(10).build()?))
    .then_store_with(qdrant.clone())
    .run()
    .await?;

    Ok(())
}
