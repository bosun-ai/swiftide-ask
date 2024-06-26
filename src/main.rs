mod config;
use std::path::PathBuf;

use anyhow::{Context, Result};
use indoc::formatdoc;
use qdrant_client::client::QdrantClientConfig;
use swiftide::{self, EmbeddingModel, SimplePrompt};

const EMBEDDING_SIZE: u64 = 1536;
// const SUPPORTED_CODE_EXTENSIONS: [&str; 27] = [
//     "py", "rs", "js", "ts", "tsx", "jsx", "vue", "go", "java", "cpp", "cxx", "hpp", "c", "swift",
//     "rb", "php", "cs", "html", "css", "sh", "kt", "clj", "cljc", "cljs", "edn", "scala", "groovy",
// ];
const SUPPORTED_CODE_EXTENSIONS: [&str; 1] = ["rs"];

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let config = config::Config::from_env();

    let path = std::env::current_dir().unwrap();

    let namespace = format!(
        "swiftide-ask-v1-{}",
        path.to_string_lossy().replace("/", "-")
    );

    let llm_client = swiftide::integrations::openai::OpenAI::builder()
        .default_embed_model("text-embedding-3-small")
        .default_prompt_model("gpt-3.5-turbo")
        .build()
        .expect("Could not build OpenAI client");

    load_codebase(config, &llm_client, &namespace, path.clone())
        .await
        .expect("Could not load documentation");

    // read question from commandline arg
    let question = std::env::args().nth(1).expect("Expected question");

    let answer = ask(config, &llm_client, &namespace, path.clone(), question)
        .await
        .expect("Could not ask question");

    println!("{}", answer);
}

async fn ask(
    config: &config::Config,
    llm_client: &swiftide::integrations::openai::OpenAI,
    namespace: &str,
    _path: PathBuf,
    question: String,
) -> Result<String> {
    let qdrant_client = QdrantClientConfig::from_url(config.qdrant_url.as_str())
        .with_api_key(config.qdrant_api_key.clone())
        .build()
        .expect("Could not build Qdrant client");

    let embedded_question_vec = llm_client.embed(vec![question.clone()]).await?;
    let embedded_question = embedded_question_vec
        .first()
        .context("Expected at least one embedding")?;

    let answer_context_points = qdrant_client
        .search_points(&qdrant_client::qdrant::SearchPoints {
            collection_name: namespace.to_string(),
            vector: embedded_question.to_owned(),
            limit: 10,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;

    let answer_context =
        answer_context_points
            .result
            .into_iter()
            .fold(String::new(), |acc, point| {
                point
                    .payload
                    .into_iter()
                    .fold(acc, |acc, (k, v)| format!("{}\n{}: {}", acc, k, v))
            });

    let prompt = formatdoc!(
        r#"
        Answer the following question(s):
        {question}

        ## Constraints
        * Only answer based on the provided context below
        * Answer the question fully and remember to be concise
        * Do not mention technologies, tools or symbols that are not present in the context

        ## Context:
        {answer_context}
        "#,
    );

    let answer = llm_client.prompt(prompt.as_str()).await?;

    Ok(answer)
}

async fn load_codebase(
    config: &config::Config,
    llm_client: &swiftide::integrations::openai::OpenAI,
    namespace: &str,
    path: PathBuf,
) -> Result<()> {
    // Load any documentation
    swiftide::ingestion::IngestionPipeline::from_loader(
        swiftide::loaders::FileLoader::new(path.clone()).with_extensions(&["md"]),
    )
    .with_concurrency(50)
    .filter_cached(swiftide::integrations::redis::Redis::try_from_url(
        config.redis_url.as_deref().context("Expected redis url")?,
        namespace,
    )?)
    .then_chunk(swiftide::transformers::ChunkMarkdown::with_chunk_range(
        100..5000,
    ))
    .then(swiftide::transformers::MetadataQAText::new(
        llm_client.clone(),
    ))
    .then_in_batch(100, swiftide::transformers::Embed::new(llm_client.clone()))
    .then_store_with(
        swiftide::integrations::qdrant::Qdrant::builder()
            .client(
                QdrantClientConfig::from_url(config.qdrant_url.as_str())
                    .with_api_key(config.qdrant_api_key.clone())
                    .build()
                    .expect("Could not build Qdrant client"),
            )
            .collection_name(namespace.to_string())
            .batch_size(1024)
            .vector_size(EMBEDDING_SIZE)
            .build()?,
    )
    .run()
    .await?;

    swiftide::ingestion::IngestionPipeline::from_loader(
        swiftide::loaders::FileLoader::new(path.clone())
            .with_extensions(&SUPPORTED_CODE_EXTENSIONS),
    )
    .with_concurrency(50)
    .filter_cached(swiftide::integrations::redis::Redis::try_from_url(
        config.redis_url.as_deref().context("Expected redis url")?,
        namespace,
    )?)
    .then_chunk(
        swiftide::transformers::ChunkCode::try_for_language_and_chunk_size("rust", 10..2048)?,
    )
    .log_errors()
    .filter_errors()
    .then(swiftide::transformers::MetadataQACode::new(
        llm_client.clone(),
    ))
    .then_in_batch(10, swiftide::transformers::Embed::new(llm_client.clone()))
    .then_store_with(
        swiftide::integrations::qdrant::Qdrant::builder()
            .client(
                QdrantClientConfig::from_url(config.qdrant_url.as_str())
                    .with_api_key(config.qdrant_api_key.clone())
                    .build()
                    .expect("Could not build Qdrant client"),
            )
            .batch_size(1024)
            .vector_size(EMBEDDING_SIZE)
            .collection_name(namespace.to_string())
            .build()?,
    )
    .run()
    .await?;

    Ok(())
}
