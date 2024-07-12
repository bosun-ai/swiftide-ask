mod config;
use std::path::PathBuf;

use anyhow::{Context, Result};
use indoc::formatdoc;
use qdrant_client::config::QdrantConfig;
use std::io::Write;
use swiftide::{self, EmbeddingModel, SimplePrompt};

const EMBEDDING_SIZE: u64 = 3072;
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
        "swiftide-ask-v3-{}",
        path.to_string_lossy().replace("/", "-")
    );

    let llm_client = swiftide::integrations::openai::OpenAI::builder()
        .default_embed_model("text-embedding-3-large")
        .default_prompt_model("gpt-3.5-turbo")
        .build()
        .expect("Could not build OpenAI client");

    load_codebase(config, &llm_client, &namespace, path.clone())
        .await
        .expect("Could not load documentation");

    let llm_client = swiftide::integrations::openai::OpenAI::builder()
        .default_embed_model("text-embedding-3-large")
        .default_prompt_model("gpt-4o")
        .build()
        .expect("Could not build OpenAI client");

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

        let answer = ask(config, &llm_client, &namespace, path.clone(), question)
            .await
            .expect("Could not ask question");
        println!("{}", answer);
    }
}

async fn ask(
    config: &config::Config,
    llm_client: &swiftide::integrations::openai::OpenAI,
    namespace: &str,
    _path: PathBuf,
    question: String,
) -> Result<String> {
    let qdrant_client = QdrantConfig::from_url(config.qdrant_url.as_str())
        .api_key(config.qdrant_api_key.clone())
        .build()
        .expect("Could not build Qdrant client");

    let transformed_question = llm_client.prompt(&formatdoc!(r"
        Your job is to help a code query tool finding the right context.

        Given the following question:
        {question}

        Please think of 5 additional questions that can help answering the original question. The code is written in {lang}.

        Especially consider what might be relevant to answer the question, like dependencies, usage and structure of the code.

        Please respond with the original question and the additional questions only.

        ## Example

        - {question}
        - Additional question 1
        - Additional question 2
        - Additional question 3
        - Additional question 4
        - Additional question 5
        ", question = question, lang = "rust"
    )).await?;

    let embedded_question = llm_client
        .embed(vec![transformed_question.clone()])
        .await?
        .pop()
        .context("Expected embedding")?;

    let answer_context_points = qdrant_client
        .search_points(
            qdrant_client::qdrant::SearchPointsBuilder::new(
                namespace.to_string(),
                embedded_question,
                10,
            )
            .with_payload(true),
        )
        .await?;

    // Probably better to return the documents themselves, not the metadata
    let answer_context = answer_context_points
        .result
        .into_iter()
        .map(|v| v.payload.get("content").unwrap().to_string())
        .collect::<Vec<_>>()
        .join("\n\n");
    // let answer_context =
    //     answer_context_points
    //         .result
    //         .into_iter()
    //         .fold(String::new(), |acc, point| {
    //             point
    //                 .payload
    //                 .into_iter()
    //                 .fold(acc, |acc, (k, v)| format!("{}\n{}: {}", acc, k, v))
    //         });

    let prompt = formatdoc!(
        r#"
        Answer the following question(s):
        {question}

        ## Constraints
        * Only answer based on the provided context below
        * Always reference files by the full path if it is relevant to the question
        * Answer the question fully and remember to be concise
        * Only answer based on the given context. If you cannot answer the question based on the
            context, say so.

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
    swiftide::ingestion::Pipeline::from_loader(
        swiftide::loaders::FileLoader::new(path.clone()).with_extensions(&["md"]),
    )
    .filter_cached(swiftide::integrations::redis::Redis::try_from_url(
        config.redis_url.as_deref().context("Expected redis url")?,
        namespace,
    )?)
    .then_chunk(swiftide::transformers::ChunkMarkdown::from_chunk_range(
        100..5000,
    ))
    .then(swiftide::transformers::MetadataQAText::new(
        llm_client.clone(),
    ))
    .then_in_batch(100, swiftide::transformers::Embed::new(llm_client.clone()))
    .then_store_with(
        swiftide::integrations::qdrant::Qdrant::builder()
            .client(
                QdrantConfig::from_url(config.qdrant_url.as_str())
                    .api_key(config.qdrant_api_key.clone())
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

    swiftide::ingestion::Pipeline::from_loader(
        swiftide::loaders::FileLoader::new(path.clone())
            .with_extensions(&SUPPORTED_CODE_EXTENSIONS),
    )
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
                QdrantConfig::from_url(config.qdrant_url.as_str())
                    .api_key(config.qdrant_api_key.clone())
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
