use std::sync::Arc;
use axum::{
    extract::{RawQuery, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde_json::json;
use tracing::error;

use crate::classifier::Classifier;
use crate::models::{ClassificationVerdict, TorrentPayload};

pub struct AppState {
    pub classifier: Arc<Classifier>,
}

pub fn create_router(classifier: Arc<Classifier>) -> Router {
    let state = Arc::new(AppState { classifier });
    Router::new()
        .route("/classify", post(classify_handler))
        .with_state(state)
}

fn parse_early_lang_query(raw_query: Option<&str>) -> Option<Vec<String>> {
    let query_str = raw_query?;
    let mut langs = Vec::new();
    for (key, val) in form_urlencoded::parse(query_str.as_bytes()) {
        if key == "early_lang" || key == "early_lang[]" {
            for part in val.replace(',', " ").split_whitespace() {
                let cleaned = part.trim().to_lowercase();
                if !cleaned.is_empty() {
                    langs.push(cleaned);
                }
            }
        }
    }
    if langs.is_empty() {
        None
    } else {
        Some(langs)
    }
}

async fn classify_handler(
    State(state): State<Arc<AppState>>,
    RawQuery(raw_query): RawQuery,
    Json(payload): Json<TorrentPayload>,
) -> Result<Json<ClassificationVerdict>, AppError> {
    let early_langs = parse_early_lang_query(raw_query.as_deref());

    let classifier = state.classifier.clone();
    // Run CPU-bound classifier in tokio blocking pool
    let verdict = tokio::task::spawn_blocking(move || {
        classifier.classify(&payload, early_langs.as_deref())
    })
    .await
    .map_err(|e| AppError(anyhow::anyhow!("Task join error: {e}")))?
    .map_err(AppError)?;

    Ok(Json(verdict))
}

pub struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        error!("Error processing request: {:?}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}
