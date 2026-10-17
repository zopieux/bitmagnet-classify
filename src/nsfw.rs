use std::fs::File;
use std::path::Path;
use std::sync::LazyLock;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use regex::Regex;
use tokenizers::TruncationParams;

use crate::text::sanitize_title_for_model;

pub const DEFAULT_THRESHOLD: f32 = 0.70;

pub const MAINSTREAM_ALLOWLIST: &[&str] = &[
    r"sex\s+(?:and\s+the\s+city|education|life|lives\s+of\s+college\s+girls|pistols)",
    r"masters\s+of\s+sex",
    r"hardcore\s+history",
    r"(?:the\s+)?big\s+bang\s+theory",
    r"kiss\s+kiss\s+bang\s+bang",
    r"bang\s+bang\s+baby",
    r"chitty\s+chitty\s+bang\s+bang",
];

static MAINSTREAM_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = format!(
        r"(?i)(?:^|[\W_])(?:{})(?:$|[\W_])",
        MAINSTREAM_ALLOWLIST.join("|")
    );
    Regex::new(&pattern).expect("Valid mainstream pattern")
});

static NON_WORD_SEP_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\W_]+").expect("Valid non-word separator pattern")
});

pub const KEYWORDS_XXX: &[&str] = &[
    r"anal",
    r"ass(?:es)?",
    r"bangbros",
    r"bdsm",
    r"blowjobs?",
    r"bondage",
    r"boob\w*",
    r"brazzers",
    r"caribbeancom",
    r"cocks?",
    r"creampie",
    r"crossdress\w*",
    r"cuck\w*",
    r"cum\w*",
    r"deepthroat",
    r"dicks?",
    r"digitaldesire",
    r"dildo",
    r"erocomic",
    r"erotic\w*",
    r"evilangel",
    r"fansly\w*",
    r"fc2",
    r"femboy",
    r"fetish",
    r"footjob",
    r"fuck\w*",
    r"gangbang",
    r"gloryhole\w*",
    r"handjob",
    r"hardcore",
    r"hentai",
    r"heyzo",
    r"incest",
    r"interracial",
    r"jav",
    r"jizz\w*",
    r"just[\s_-]?for[\s_-]?fans?",
    r"kink\w*",
    r"ladyboy",
    r"legalporno",
    r"lesbian[sx]?",
    r"lewd\w*",
    r"masturbat\w*",
    r"milf\w*",
    r"naughtyamerica",
    r"netorare",
    r"nubile\w*",
    r"only[\s_-]?fans?",
    r"orgasm\w*",
    r"orgy",
    r"panties",
    r"panty",
    r"pegging",
    r"playboy",
    r"playmate",
    r"porn\w*",
    r"pov",
    r"pthc",
    r"ptsc",
    r"pussy",
    r"realitykings",
    r"seduc\w*",
    r"sex\w*",
    r"shemale",
    r"sissy",
    r"slut\w*",
    r"squirt\w*",
    r"step[\s_-]?(?:daughter|mom|sister|son|dad|father|mother)",
    r"strapon",
    r"stripper",
    r"submissive",
    r"succubus",
    r"swinger",
    r"taboo",
    r"teamskeet",
    r"threesome",
    r"tits?",
    r"tonguejob",
    r"topless",
    r"transsexual",
    r"twistys",
    r"vibrator",
    r"wank\w*",
    r"xxx",
];

static TOKEN_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = format!(r"(?i)(?:^|[\W_])({})(?:$|[\W_])", KEYWORDS_XXX.join("|"));
    Regex::new(&pattern).expect("Valid token pattern")
});

static WILDCARD_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:fuck|gloryhole|porn|wank|xxx)").expect("Valid wildcard pattern")
});

pub fn is_mainstream_allowed(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let normalized = NON_WORD_SEP_PATTERN.replace_all(text, " ");
    MAINSTREAM_PATTERN.is_match(&normalized)
}

pub fn is_keyword_nsfw(text: &str) -> bool {
    if text.is_empty() || is_mainstream_allowed(text) {
        return false;
    }
    TOKEN_PATTERN.is_match(text) || WILDCARD_PATTERN.is_match(text)
}

pub struct NsfwModel {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl NsfwModel {
    pub fn load(model_dir: &Path, dylib_path: Option<&str>, intra_threads: usize) -> Result<Self> {
        let onnx_path = model_dir.join("backbone_int8.onnx");
        let weights_path = model_dir.join("head_weights.npz");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !onnx_path.is_file() {
            anyhow::bail!("Missing ONNX model at {:?}", onnx_path);
        }
        if !weights_path.is_file() {
            anyhow::bail!("Missing head weights at {:?}", weights_path);
        }
        if !tokenizer_path.is_file() {
            anyhow::bail!("Missing tokenizer at {:?}", tokenizer_path);
        }

        // Initialize ONNX Runtime environment
        let env = match dylib_path {
            Some(path) => ort::init_from(path)
                .context("Failed to init ONNX Runtime from dynamic library path")?
                .with_name("bitmagnet-classify")
                .build()
                .context("Failed to build ORT environment")?,
            None => ort::init()
                .with_name("bitmagnet-classify")
                .build()
                .context("Failed to build default ORT environment")?,
        };

        let session = Session::builder(&env)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(&onnx_path)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 96,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut npz = NpzReader::new(File::open(&weights_path)?)?;
        let w1: Array2<f32> = npz.by_name("W1.npy")?;
        let b1: Array1<f32> = npz.by_name("b1.npy")?;
        let w2: Array2<f32> = npz.by_name("W2.npy")?;
        let b2: Array1<f32> = npz.by_name("b2.npy")?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            w1,
            b1,
            w2,
            b2,
        })
    }

    pub fn is_model_nsfw(&self, text: &str, threshold: f32) -> Result<(bool, f32)> {
        if text.trim().is_empty() || is_mainstream_allowed(text) {
            return Ok((false, 0.0));
        }

        let sanitized = sanitize_title_for_model(text);
        if sanitized.is_empty() {
            return Ok((false, 0.0));
        }

        let prompt = format!("query: {sanitized}");
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let seq_len = input_ids.len();

        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)?;
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)?;

        let input_ids_tensor = Tensor::from_array(input_ids_array)?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array.clone())?;

        let mut session = self.session.lock();
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])?;

        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let last_hidden = ndarray::ArrayView3::from_shape((1, seq_len, 384), data)?;

        // Mean pooling + L2 normalization
        let mut sum_embeddings = Array1::<f32>::zeros(384);
        let mut sum_mask = 0.0f32;
        for (i, &m) in attention_mask_array.row(0).iter().enumerate() {
            let m_f = m as f32;
            sum_mask += m_f;
            for d in 0..384 {
                sum_embeddings[d] += last_hidden[[0, i, d]] * m_f;
            }
        }
        let divisor = sum_mask.max(1e-9);
        let mut pooled = sum_embeddings / divisor;
        let norm = (pooled.dot(&pooled)).sqrt().max(1e-12);
        pooled /= norm;

        // 2-layer MLP head: h = max(0, pooled @ W1.T + b1)
        let mut h = Array1::<f32>::zeros(128);
        for j in 0..128 {
            let mut val = self.b1[j];
            for k in 0..384 {
                val += self.w1[[j, k]] * pooled[k];
            }
            h[j] = val.max(0.0);
        }

        // logits = h @ W2.T + b2
        let mut logit = self.b2[0];
        for j in 0..128 {
            logit += self.w2[[0, j]] * h[j];
        }

        let prob = 1.0 / (1.0 + (-logit).exp());
        Ok((prob >= threshold, prob))
    }

    pub fn check_nsfw(&self, text: &str, threshold: f32) -> Result<(bool, &'static str)> {
        if is_mainstream_allowed(text) {
            return Ok((false, ""));
        }
        if is_keyword_nsfw(text) {
            return Ok((true, "keyword"));
        }
        let (nsfw, _) = self.is_model_nsfw(text, threshold)?;
        if nsfw {
            return Ok((true, "model"));
        }
        Ok((false, ""))
    }
}
