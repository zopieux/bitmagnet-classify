use anyhow::{Context, Result};
use reqwest::Client;
use std::time::Duration;

use crate::models::{
    ApiClassificationItem, ApiGetBatchResponse, ApiPatchBatchResponse, PatchClassificationItem,
};

#[derive(Clone, Debug)]
pub struct BitmagnetClient {
    client: Client,
    endpoint: String,
    auth: Option<(String, Option<String>)>,
}

impl BitmagnetClient {
    pub fn new(raw_endpoint: &str, user_auth: Option<&str>) -> Result<Self> {
        let mut parsed_url = reqwest::Url::parse(raw_endpoint.trim())
            .with_context(|| format!("Invalid endpoint URL: {raw_endpoint}"))?;

        let url_user = if !parsed_url.username().is_empty() {
            let u = parsed_url.username().to_string();
            let p = parsed_url.password().map(|s| s.to_string());
            Some((u, p))
        } else {
            None
        };

        let auth = if let Some(user_str) = user_auth.filter(|s| !s.is_empty()) {
            let (u, p) = match user_str.split_once(':') {
                Some((u, p)) => (u.to_string(), Some(p.to_string())),
                None => (user_str.to_string(), None),
            };
            Some((u, p))
        } else {
            url_user
        };

        // Clear username & password from URL so it doesn't leak into logs
        let _ = parsed_url.set_username("");
        let _ = parsed_url.set_password(None);

        let mut path = parsed_url.path().trim_end_matches('/').to_string();
        if !path.ends_with("/api/v1/classification") {
            if path.is_empty() {
                path = "/api/v1/classification".to_string();
            } else {
                path = format!("{path}/api/v1/classification");
            }
        }
        parsed_url.set_path(&path);
        let endpoint = parsed_url.to_string();

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            client,
            endpoint,
            auth,
        })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn has_auth(&self) -> bool {
        self.auth.is_some()
    }

    /// Fetches a batch of torrents to classify.
    /// If `exclude_tag` is Some(tag), passes `?exclude_tag={tag}`.
    /// If `tag` is Some(tag), passes `?tag={tag}`.
    /// `limit` sets the max items per batch.
    pub async fn get_batch(
        &self,
        limit: usize,
        exclude_tag: Option<&str>,
        tag: Option<&str>,
    ) -> Result<Vec<ApiClassificationItem>> {
        let mut req = self
            .client
            .get(&self.endpoint)
            .query(&[("limit", limit.to_string())]);

        if let Some((user, pass)) = &self.auth {
            req = req.basic_auth(user, pass.as_deref());
        }

        if let Some(et) = exclude_tag.filter(|s| !s.is_empty()) {
            req = req.query(&[("exclude_tag", et)]);
        }

        if let Some(t) = tag.filter(|s| !s.is_empty()) {
            req = req.query(&[("tag", t)]);
        }

        let resp = req
            .send()
            .await
            .with_context(|| format!("Failed to GET from bitmagnet at {}", self.endpoint))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("GET {} returned status {}: {}", self.endpoint, status, body);
        }

        let batch: ApiGetBatchResponse = resp
            .json()
            .await
            .with_context(|| format!("Failed to parse JSON response from {}", self.endpoint))?;

        Ok(batch.items)
    }

    /// Sends a batch of classification updates to bitmagnet.
    ///
    /// If the server rejects the batch because a tag references a torrent that
    /// no longer exists, the batch is split and retried so that one stale item
    /// cannot block the rest of the batch.
    pub async fn patch_batch(&self, items: &[PatchClassificationItem]) -> Result<usize> {
        let mut pending: Vec<&[PatchClassificationItem]> = vec![items];
        let mut updated = 0;

        while let Some(chunk) = pending.pop() {
            match self.patch_batch_once(chunk).await {
                Ok(count) => updated += count,
                Err(error) if is_foreign_key_error(&error) => {
                    if chunk.len() <= 1 {
                        if let Some(item) = chunk.first() {
                            tracing::warn!(
                                "Skipping patch for {} after foreign key error: {:#}",
                                item.info_hash,
                                error
                            );
                        }
                    } else {
                        let mid = chunk.len() / 2;
                        let (left, right) = chunk.split_at(mid);
                        pending.push(left);
                        pending.push(right);
                    }
                }
                Err(error) => return Err(error),
            }
        }

        Ok(updated)
    }

    async fn patch_batch_once(&self, items: &[PatchClassificationItem]) -> Result<usize> {
        if items.is_empty() {
            return Ok(0);
        }

        let mut req = self.client.patch(&self.endpoint).json(items);

        if let Some((user, pass)) = &self.auth {
            req = req.basic_auth(user, pass.as_deref());
        }

        let resp = req
            .send()
            .await
            .with_context(|| format!("Failed to PATCH bitmagnet at {}", self.endpoint))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "PATCH {} returned status {}: {}",
                self.endpoint,
                status,
                body
            );
        }

        let res: ApiPatchBatchResponse = resp
            .json()
            .await
            .with_context(|| format!("Failed to parse PATCH response from {}", self.endpoint))?;

        Ok(res.updated)
    }
}

fn is_foreign_key_error(error: &anyhow::Error) -> bool {
    format!("{error:#}")
        .to_ascii_lowercase()
        .contains("foreign key")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_foreign_key_errors() {
        let error = anyhow::anyhow!(
            "PATCH http://localhost/api/v1/classification returned status 500: {{\"error\":\"constraint failed: FOREIGN KEY constraint failed (787)\"}}"
        );
        assert!(is_foreign_key_error(&error));
        assert!(!is_foreign_key_error(&anyhow::anyhow!(
            "connection refused"
        )));
    }
}
