use std::env;
use std::sync::OnceLock;

static CONFIG: OnceLock<Config> = OnceLock::new();

#[allow(dead_code)]
pub struct Config {
    pub qdrant_url: String,
    pub qdrant_api_key: Option<String>,
    pub redis_url: Option<String>,
}

impl Config {
    pub fn from_env() -> &'static Config {
        CONFIG.get_or_init(|| {
            let qdrant_url = env::var("QDRANT_URL").expect("QDRANT_URL env var not set");
            let qdrant_api_key = env::var("QDRANT_API_KEY").ok();
            let redis_url = env::var("REDIS_URL").ok();

            Self {
                qdrant_url,
                qdrant_api_key,
                redis_url,
            }
        })
    }
}
