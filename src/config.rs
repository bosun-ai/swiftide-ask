use std::env;
use std::sync::OnceLock;

static CONFIG: OnceLock<Config> = OnceLock::new();

#[allow(dead_code)]
pub struct Config {
    pub openai_api_key: String,
    pub openai_endpoint: Option<String>,
    pub qdrant_url: String,
    pub qdrant_api_key: Option<String>,
    pub redis_url: Option<String>,
}

impl Config {
    pub fn from_env() -> &'static Config {
        CONFIG.get_or_init(|| {
            let openai_api_key =
                env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var not set");
            let openai_endpoint = env::var("OPENAI_ENDPOINT").ok();
            let qdrant_url = env::var("QDRANT_URL").expect("QDRANT_URL env var not set");
            let qdrant_api_key = env::var("QDRANT_API_KEY").ok();
            let redis_url = env::var("REDIS_URL").ok();

            Self {
                openai_api_key,
                openai_endpoint,
                qdrant_url,
                qdrant_api_key,
                redis_url,
            }
        })
    }
}
