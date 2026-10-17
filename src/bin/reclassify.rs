use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;

use bitmagnet_classify::classifier::Classifier;
use bitmagnet_classify::langid::LanguageDetector;
use bitmagnet_classify::models::{TorrentFile, TorrentPayload};
use bitmagnet_classify::nsfw::NsfwModel;

#[derive(Parser, Debug)]
#[command(
    name = "reclassify",
    about = "Batch reclassify torrent titles and files from CSV"
)]
struct Args {
    #[arg(default_value = "scratch/titles_with_files.csv")]
    input_csv: PathBuf,

    #[arg(short, long, default_value = "scratch/classified_rust.csv")]
    output: PathBuf,

    #[arg(short, long, default_value_t = 4)]
    threads: usize,

    #[arg(short, long)]
    limit: Option<usize>,

    #[arg(
        long,
        value_delimiter = ' ',
        num_args = 0..,
        default_values = ["ja", "ko", "zh", "ru", "uk", "be", "bg", "sr", "mk"]
    )]
    early_lang: Vec<String>,

    #[arg(long, env = "NSFW_ONNX_PATH")]
    model_dir: Option<PathBuf>,

    #[arg(long, env = "FASTTEXT_MODEL_PATH")]
    fasttext_model_path: Option<PathBuf>,

    #[arg(long, env = "ORT_DYLIB_PATH")]
    ort_dylib_path: Option<String>,
}

struct TorrentItem {
    name: String,
    sample_paths: String,
    extensions: String,
}

impl TorrentItem {
    fn to_payload(&self) -> TorrentPayload {
        let mut files = Vec::new();
        if !self.sample_paths.is_empty() {
            for p in self.sample_paths.split('|') {
                let p = p.trim();
                if p.is_empty() {
                    continue;
                }
                let ext = if let Some((_, ext)) = p.rsplit_once('.') {
                    Some(ext.to_lowercase())
                } else {
                    None
                };
                files.push(TorrentFile {
                    path: p.to_string(),
                    size: 0,
                    extension: ext,
                });
            }
        }

        let first_ext = if !self.extensions.is_empty() {
            let ext = self.extensions.split(',').next().unwrap_or("").trim();
            if ext.is_empty() {
                None
            } else {
                Some(ext.to_string())
            }
        } else {
            None
        };

        TorrentPayload {
            info_hash: "0000000000000000000000000000000000000000".to_string(),
            name: self.name.clone(),
            size: 0,
            extension: first_ext,
            files,
            hint: None,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let nsfw_model = if let Some(ref dir) = args.model_dir {
        if !dir.exists() {
            anyhow::bail!("NSFW model directory does not exist: {:?}", dir);
        }
        tracing::info!("NSFW model: enabled ({:?})", dir);
        Some(
            NsfwModel::load(
                dir,
                args.ort_dylib_path.as_deref(),
                args.threads,
            )
            .context("Failed to load NSFW ONNX model")?,
        )
    } else {
        tracing::info!("NSFW model: disabled (keyword filter only)");
        None
    };

    let lang_detector = if let Some(ref path) = args.fasttext_model_path {
        if !path.exists() {
            anyhow::bail!("FastText model file does not exist: {:?}", path);
        }
        tracing::info!("Language model: fastText enabled ({:?})", path);
        LanguageDetector::load_fasttext(path, bitmagnet_classify::langid::DEFAULT_THRESHOLD)?
    } else {
        tracing::info!("Language model: whatlang (model unset, using cheap fallback)");
        LanguageDetector::with_whatlang(bitmagnet_classify::langid::DEFAULT_THRESHOLD)
    };

    let early_lang_set: HashSet<String> = args
        .early_lang
        .iter()
        .flat_map(|s| s.replace(',', " ").split_whitespace().map(|x| x.trim().to_lowercase()).collect::<Vec<_>>())
        .filter(|s| !s.is_empty())
        .collect();

    let classifier = Arc::new(Classifier::new(
        lang_detector,
        nsfw_model,
        early_lang_set.clone(),
    ));

    println!("Loading items from {:?}", args.input_csv);
    let in_file = File::open(&args.input_csv)
        .with_context(|| format!("Failed to open input file {:?}", args.input_csv))?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(in_file);

    let mut items = Vec::new();
    for result in reader.records() {
        let record = result?;
        if record.is_empty() {
            continue;
        }
        let name = record.get(0).unwrap_or("").trim().to_string();
        let sample_paths = record.get(1).unwrap_or("").trim().to_string();
        let extensions = record.get(2).unwrap_or("").trim().to_string();

        items.push(TorrentItem {
            name,
            sample_paths,
            extensions,
        });

        if let Some(limit) = args.limit {
            if items.len() >= limit {
                break;
            }
        }
    }

    let total = items.len();
    println!("Loaded {total} items from {:?}", args.input_csv);
    println!("Early-return languages ({}): {:?}", early_lang_set.len(), early_lang_set);
    println!("Running with Rayon thread pool ({} threads)...", args.threads);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .context("Failed to build Rayon thread pool")?;

    let out_file = File::create(&args.output)
        .with_context(|| format!("Failed to create output file {:?}", args.output))?;
    let mut writer = csv::WriterBuilder::new().from_writer(out_file);

    let t0 = Instant::now();
    let early_lang_vec: Vec<String> = early_lang_set.into_iter().collect();

    let chunk_size = 500;
    let mut processed = 0;

    for chunk in items.chunks(chunk_size) {
        let results: Vec<(String, &TorrentItem)> = pool.install(|| {
            chunk
                .par_iter()
                .map(|item| {
                    let payload = item.to_payload();
                    let (action, tag_str) = match classifier.classify(&payload, Some(&early_lang_vec)) {
                        Ok(verdict) => {
                            let tag_str = verdict.tags.join(";");
                            (verdict.action.to_string(), tag_str)
                        }
                        Err(e) => ("error".to_string(), format!("error:{e}")),
                    };

                    let verdict_col = if tag_str.is_empty() {
                        action
                    } else {
                        format!("{action}:{tag_str}")
                    };

                    (verdict_col, item)
                })
                .collect()
        });

        for (verdict_col, item) in results {
            writer.write_record(&[&verdict_col, &item.name, &item.sample_paths])?;
        }

        processed += chunk.len();
        let elapsed = t0.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 { processed as f64 / elapsed } else { 0.0 };
        let pct = (processed * 100) / total;
        println!("Progress: {processed}/{total} ({pct}%) - {rate:.0} items/s");
    }

    writer.flush()?;
    let elapsed = t0.elapsed().as_secs_f64();
    let rate = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };
    println!("\nFinished writing {total} classified rows to {:?} in {elapsed:.1}s ({rate:.0} items/s)", args.output);

    Ok(())
}
