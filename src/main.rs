use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use tracing::info;

use bitmagnet_classify::classifier::Classifier;
use bitmagnet_classify::client::BitmagnetClient;
use bitmagnet_classify::langid::LanguageDetector;
use bitmagnet_classify::models::{
    is_valid_tag_name, PatchClassificationItem, VerdictAction,
};
use bitmagnet_classify::nsfw::NsfwModel;

#[derive(Parser, Debug)]
#[command(
    name = "bitmagnet-classify",
    about = "Online batch classifier talking to bitmagnet classification API"
)]
struct Args {
    #[arg(
        value_name = "ENDPOINT",
        help = "Bitmagnet API endpoint URL (e.g. http://localhost:9092)",
        default_value = "http://localhost:9092",
        env = "BITMAGNET_URL"
    )]
    endpoint: String,

    #[arg(
        long = "tag",
        default_value = "ok",
        env = "FILTER_TAG",
        help = "Classification indicator tag: checked for absence when GETting, and added to PATCHed tags"
    )]
    tag: String,

    #[arg(
        long = "query-tag",
        help = "Only query torrents that already have this tag present"
    )]
    query_tag: Option<String>,

    #[arg(
        long,
        help = "Reprocess all torrents ignoring the filter tag (cannot be combined with --tag exclusion)"
    )]
    reprocess_all: bool,

    #[arg(long, default_value_t = 50, env = "BATCH_SIZE", help = "Batch size for reading and writing torrents")]
    batch_size: usize,

    #[arg(long, default_value_t = 1000, env = "RATE", help = "Maximum processing rate in torrents per second")]
    rate: u32,

    #[arg(short, long, env = "THREADS", help = "Number of worker threads for parallel classification")]
    threads: Option<usize>,

    #[arg(short, long, env = "LIMIT", help = "Maximum number of torrents to process in total")]
    limit: Option<usize>,

    #[arg(
        long,
        help = "Watch mode: poll continuously every N seconds when no new torrents are available"
    )]
    poll_interval: Option<u64>,

    #[arg(
        long,
        value_delimiter = ' ',
        num_args = 0..,
        default_values = ["ja", "ko", "zh", "ru", "uk", "be", "bg", "sr", "mk"],
        env = "EARLY_LANG"
    )]
    early_lang: Vec<String>,

    #[arg(long, env = "NSFW_ONNX_PATH", help = "Path to NSFW ONNX model directory")]
    model_dir: Option<PathBuf>,

    #[arg(long, env = "FASTTEXT_MODEL_PATH", help = "Path to FastText language model file")]
    fasttext_model_path: Option<PathBuf>,

    #[arg(long, env = "ORT_DYLIB_PATH", help = "Path to ONNX Runtime shared library")]
    ort_dylib_path: Option<String>,

    #[arg(long, default_value_t = 50, help = "Max files per torrent to evaluate")]
    max_files_per_torrent: usize,

    #[arg(
        short = 'u',
        long = "user",
        help = "Basic auth credentials in format 'user:password' (like curl)",
        env = "BITMAGNET_USER"
    )]
    user: Option<String>,

    #[arg(
        long,
        help = "Dry run mode: classify and print formatted PATCH JSON to stdout without sending updates to the API"
    )]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    if !is_valid_tag_name(&args.tag) {
        anyhow::bail!(
            "Invalid tag {:?}: must be lowercase alphanumeric with optional hyphens (matching '^[a-z0-9]+(-[a-z0-9]+)*$')",
            args.tag
        );
    }

    let exclude_tag = if args.reprocess_all {
        None
    } else {
        Some(args.tag.as_str())
    };

    let client = BitmagnetClient::new(&args.endpoint, args.user.as_deref())
        .with_context(|| format!("Failed to initialize client for endpoint {}", args.endpoint))?;

    let num_threads = args
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));

    info!("Bitmagnet endpoint: {}", client.endpoint());
    if client.has_auth() {
        info!("Authentication: Basic auth enabled");
    }
    info!("Threads: {}", num_threads);
    info!("Batch size: {}", args.batch_size);
    info!("Max rate: {} items/s", args.rate);
    match exclude_tag {
        Some(tag) => info!("Tag indicator: {:?} (exclude_tag on GET, added on PATCH)", tag),
        None => info!("Mode: reprocess all torrents"),
    }
    if let Some(ref qt) = args.query_tag {
        info!("Query tag filter: {:?}", qt);
    }
    if let Some(limit) = args.limit {
        info!("Limit: {} torrents", limit);
    }
    if let Some(interval) = args.poll_interval {
        info!("Watch mode enabled: poll interval {}s", interval);
    }
    if args.dry_run {
        info!("Dry run mode: enabled (no PATCH requests will be sent)");
    }

    // Initialize ML / NLP models
    let nsfw_model = if let Some(ref dir) = args.model_dir {
        if !dir.exists() {
            anyhow::bail!("NSFW model directory does not exist: {:?}", dir);
        }
        info!("NSFW model: enabled ({:?})", dir);
        Some(
            NsfwModel::load(dir, args.ort_dylib_path.as_deref(), num_threads)
                .context("Failed to load NSFW ONNX model")?,
        )
    } else {
        info!("NSFW model: disabled (keyword filter only)");
        None
    };

    let lang_detector = if let Some(ref path) = args.fasttext_model_path {
        if !path.exists() {
            anyhow::bail!("FastText model file does not exist: {:?}", path);
        }
        info!("Language model: fastText enabled ({:?})", path);
        LanguageDetector::load_fasttext(path, bitmagnet_classify::langid::DEFAULT_THRESHOLD)?
    } else {
        info!("Language model: whatlang (model unset, using fallback)");
        LanguageDetector::with_whatlang(bitmagnet_classify::langid::DEFAULT_THRESHOLD)
    };

    let early_lang_set: HashSet<String> = args
        .early_lang
        .iter()
        .flat_map(|s| {
            s.replace(',', " ")
                .split_whitespace()
                .map(|x| x.trim().to_lowercase())
                .collect::<Vec<_>>()
        })
        .filter(|s| !s.is_empty())
        .collect();

    let early_lang_vec: Vec<String> = early_lang_set.iter().cloned().collect();

    let classifier = Arc::new(Classifier::new(
        lang_detector,
        nsfw_model,
        early_lang_set,
    ));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .context("Failed to build Rayon thread pool")?;

    // Setup graceful shutdown signal
    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        tokio::spawn(async move {
            if let Ok(()) = tokio::signal::ctrl_c().await {
                eprintln!("\nReceived Ctrl+C, finishing current batch and shutting down...");
                r.store(false, Ordering::SeqCst);
            }
        });
    }

    let overall_start = Instant::now();
    let mut total_processed: usize = 0;
    let mut total_updated: usize = 0;
    let mut batch_num: usize = 0;

    eprintln!("Starting online classification against bitmagnet API...");
    eprintln!("Endpoint: {}", client.endpoint());
    if let Some(tag) = exclude_tag {
        eprintln!("Classified tag: '{}'", tag);
    } else {
        eprintln!("Mode: reprocess all torrents");
    }

    while running.load(Ordering::SeqCst) {
        let batch_start = Instant::now();

        let batch_limit = if let Some(limit) = args.limit {
            if total_processed >= limit {
                break;
            }
            std::cmp::min(args.batch_size, limit - total_processed)
        } else {
            args.batch_size
        };

        // 1. Fetch batch from API
        let items = match client
            .get_batch(batch_limit, exclude_tag, args.query_tag.as_deref())
            .await
        {
            Ok(items) => items,
            Err(e) => {
                tracing::error!("Failed to fetch batch from API: {:#}", e);
                if let Some(interval) = args.poll_interval {
                    tokio::time::sleep(Duration::from_secs(interval)).await;
                    continue;
                } else {
                    return Err(e);
                }
            }
        };

        if items.is_empty() {
            if total_processed == 0 {
                eprintln!("No torrents found needing classification. Nothing to do.");
            } else {
                eprintln!("No more unclassified torrents remaining.");
            }

            if let Some(interval) = args.poll_interval {
                tokio::time::sleep(Duration::from_secs(interval)).await;
                continue;
            } else {
                break;
            }
        }

        let batch_count = items.len();
        batch_num += 1;

        // 2. Classify items in parallel
        let classifier_ref = classifier.clone();
        let early_langs_ref = &early_lang_vec;
        let tag_ref = &args.tag;
        let max_files = args.max_files_per_torrent;

        let patch_items: Vec<PatchClassificationItem> = pool.install(|| {
            items
                .par_iter()
                .map(|item| {
                    let mut payload = item.to_payload();
                    if payload.files.len() > max_files {
                        payload.files.truncate(max_files);
                    }

                    let verdict = classifier_ref
                        .classify(&payload, Some(early_langs_ref), Some(tag_ref))
                        .unwrap_or_default();

                    let action = match verdict.action {
                        VerdictAction::Delete => "delete".to_string(),
                        _ => "do_nothing".to_string(),
                    };

                    PatchClassificationItem {
                        info_hash: item.info_hash.clone(),
                        action,
                        tags: verdict.tags,
                        content_type: verdict.content_type,
                    }
                })
                .collect()
        });

        // 3. Send updates back to API (or print formatted JSON to stdout if dry-run)
        let updated_count = if args.dry_run {
            let formatted = serde_json::to_string_pretty(&patch_items)
                .context("Failed to format patch items as JSON")?;
            println!("{formatted}");
            batch_count
        } else {
            match client.patch_batch(&patch_items).await {
                Ok(count) => count,
                Err(e) => {
                    tracing::error!("Failed to patch batch in API: {:#}", e);
                    return Err(e);
                }
            }
        };

        total_processed += batch_count;
        total_updated += updated_count;

        let batch_elapsed = batch_start.elapsed();
        let overall_elapsed = overall_start.elapsed().as_secs_f64();
        let current_rate = if overall_elapsed > 0.0 {
            total_processed as f64 / overall_elapsed
        } else {
            0.0
        };

        eprintln!(
            "Batch #{}: processed {} torrents in {:.2?} (total: {}, rate: {:.1} items/s)",
            batch_num, batch_count, batch_elapsed, total_processed, current_rate
        );

        // Rate limiting
        if args.rate > 0 {
            let target_duration = Duration::from_secs_f64(batch_count as f64 / args.rate as f64);
            if batch_elapsed < target_duration {
                tokio::time::sleep(target_duration - batch_elapsed).await;
            }
        }
    }

    let elapsed = overall_start.elapsed().as_secs_f64();
    let avg_rate = if elapsed > 0.0 {
        total_processed as f64 / elapsed
    } else {
        0.0
    };

    eprintln!("\nClassification complete!");
    eprintln!(
        "Processed: {} torrents (updated: {}) in {:.2}s ({:.1} torrents/s)",
        total_processed, total_updated, elapsed, avg_rate
    );

    Ok(())
}
