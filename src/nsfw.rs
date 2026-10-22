use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use regex::Regex;
use std::fs::File;
use std::path::Path;
use std::sync::LazyLock;
use tokenizers::TruncationParams;

use crate::text;

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

static NON_WORD_SEP_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\W_]+").expect("Valid non-word separator pattern"));

pub const KEYWORDS_XXX: &[&str] = &[
    r"anal",
    r"ass(?:es)?",
    r"bang(?:bros|ing)",
    r"bdsm",
    r"belami",
    r"blowjobs?",
    r"bondage",
    r"boob\w*",
    r"brazzers",
    r"caribbeancom",
    r"chaturbate",
    r"cocks?",
    r"creampie",
    r"crossdress\w*",
    r"cuck\w*",
    r"cum\w*",
    r"deepthroat",
    r"dicks?",
    r"digitaldesire",
    r"dildo",
    r"erito",
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
    r"mature4k",
    r"manyvids",
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
    r"realjamvr",
    r"porn\w*",
    r"pov",
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
    r"titties",
    r"tonguejob",
    r"topless",
    r"tokyo[\s_-]?hot",
    r"tushy",
    r"transsexual",
    r"twistys",
    r"vibrator",
    r"wank\w*",
    r"x[-\s_]?art",
    r"xxx",
];

static TOKEN_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = format!(r"(?i)(?:^|[\W_])({})(?:$|[\W_])", KEYWORDS_XXX.join("|"));
    Regex::new(&pattern).expect("Valid token pattern")
});

static WILDCARD_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:fuck|gloryhole|porn|wank|xxx)").expect("Valid wildcard pattern")
});

pub const KEYWORDS_SUBSTRING: &[&str] = &[
    r"色情",
    r"情色",
    r"成人视频",
    r"成人影片",
    r"成人片",
    r"18禁",
    r"女优",
    r"女優",
    r"无码",
    r"無碼",
    r"无修正",
    r"無修正",
    r"内射",
    r"中出",
    r"中出し",
    r"口交",
    r"肛交",
    r"自慰",
    r"潮喷",
    r"潮吹",
    r"潮吹き",
    r"肉便器",
    r"肉棒",
    r"假阳具",
    r"淫",
    r"风骚",
    r"骚货",
    r"约炮",
    r"约啪",
    r"嫖娼",
    r"乱伦",
    r"迷奸",
    r"强奸",
    r"人妻",
    r"巨乳",
    r"爆乳",
    r"美乳",
    r"痴女",
    r"熟女",
    r"露出",
    r"偷拍",
    r"盗撮",
    r"里番",
    r"工口",
    r"嫩模",
    r"外围女",
    r"楼凤",
    r"风俗",
    r"風俗",
    r"ソープ",
    r"フェラ",
    r"セックス",
    r"ヌード",
    r"エロ",
    r"官能",
    r"乱交",
    r"痴漢",
    r"おっぱい",
    r"絶頂",
    r"アクメ",
    r"オナニー",
    r"性交",
    r"性愛",
    r"做爱",
    r"性爱",
    r"车震",
    r"开房",
    r"东热",
    r"黑丝",
    r"无套",
    r"大奶",
];

static SUBSTRING_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = format!(r"(?i)(?:{})", KEYWORDS_SUBSTRING.join("|"));
    Regex::new(&pattern).expect("Valid NSFW substring pattern")
});

const JAV_LABEL_PREFIXES: &[&str] = &[
    "abp", "ap", "bn", "carib", "cawd", "dvdms", "fc2", "fsdss", "fset", "gachinco", "gapl",
    "heyzo", "hodv", "homa", "ipx", "ipz", "jav", "jul", "jux", "kaad", "luxu", "mbr", "miaa",
    "miad", "mide", "midv", "mikr", "mium", "mmc", "nass", "nkd", "onsg", "pred", "rbd", "sdnm",
    "snis", "sone", "snos", "ssis", "ssni", "start", "sun", "tek", "venu", "waaa",
];

const JAV_CODE_EXCLUDED_PREFIXES: &[&str] = &[
    "aac",
    "ac",
    "amd",
    "audi",
    "avc",
    "bd",
    "benz",
    "bmw",
    "canon",
    "cd",
    "cpu",
    "ddp",
    "ddr",
    "dts",
    "dvd",
    "fhd",
    "ford",
    "fuji",
    "gpu",
    "gtx",
    "hdd",
    "hd",
    "hdr",
    "hevc",
    "honda",
    "hyundai",
    "intel",
    "ios",
    "iso",
    "kia",
    "mazda",
    "mlb",
    "mp",
    "mpeg",
    "nba",
    "nfl",
    "nhl",
    "nissan",
    "nvidia",
    "nvme",
    "olympus",
    "os",
    "panasonic",
    "pci",
    "psp",
    "rar",
    "rtx",
    "rx",
    "sata",
    "sdr",
    "sony",
    "ssd",
    "subaru",
    "tesla",
    "toyota",
    "ts",
    "ufc",
    "uhd",
    "usb",
    "vga",
    "vw",
    "web",
    "win",
    "windows",
    "wwe",
    "xbox",
    "zip",
];

const JAV_CODE_EXCLUDED_NUMBERS: &[&str] =
    &["480", "540", "576", "720", "1080", "1440", "2160", "4320"];

static JAV_CODE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[\W_])([a-z]{2,6})-(\d{3,6})(?:$|[\W_])").expect("Valid JAV code pattern")
});

static JAV_COMPACT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[\W_])([a-z]{2,6})(\d{3,5})(?:$|[\W_])")
        .expect("Valid compact JAV code pattern")
});

fn is_jav_code(text: &str) -> bool {
    for captures in JAV_CODE_PATTERN.captures_iter(text) {
        let prefix = captures[1].to_lowercase();
        let number = &captures[2];
        if JAV_CODE_EXCLUDED_PREFIXES.contains(&prefix.as_str())
            || JAV_CODE_EXCLUDED_NUMBERS.contains(&number)
        {
            continue;
        }
        if number.len() == 4
            && let Ok(year) = number.parse::<u32>()
            && (1900..=2099).contains(&year)
        {
            continue;
        }
        return true;
    }

    for captures in JAV_COMPACT_PATTERN.captures_iter(text) {
        let prefix = captures[1].to_lowercase();
        if JAV_LABEL_PREFIXES.contains(&prefix.as_str()) {
            return true;
        }
    }

    false
}

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
    TOKEN_PATTERN.is_match(text)
        || WILDCARD_PATTERN.is_match(text)
        || SUBSTRING_PATTERN.is_match(text)
        || is_jav_code(text)
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

        let sanitized = text::sanitize_title_for_model(text);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_adult_studios_and_codes() {
        assert!(is_keyword_nsfw("Mature4k - Sweet Becky"));
        assert!(is_keyword_nsfw("[Tushy] Kelly Collins"));
        assert!(is_keyword_nsfw("RealJamVR-Yhivi-Strip-Show"));
        assert!(is_keyword_nsfw("[BelAmi] Pillow Talk 3"));
        assert!(is_keyword_nsfw("x-art_lia_black_&_white_1080"));
        assert!(is_keyword_nsfw("Tokyo Hot nkd-064"));
        assert!(is_keyword_nsfw("ABP-449.avi"));
        assert!(is_keyword_nsfw("MIAD-923"));
        assert!(is_keyword_nsfw("FSET299.avi"));
        assert!(is_keyword_nsfw("风骚的家庭教师"));
        assert!(is_keyword_nsfw("东热收藏之靓女"));
    }

    #[test]
    fn ignores_technical_codes_and_years() {
        assert!(!is_keyword_nsfw("Windows 11 24H2 + RTX-4090"));
        assert!(!is_keyword_nsfw("The Big Short (2015) 1080p"));
        assert!(!is_keyword_nsfw("histoires-du-cinema-jlg-1988"));
        assert!(!is_keyword_nsfw("Something 2160p HEVC-10"));
        assert!(!is_keyword_nsfw("Software build Eagle123"));
    }
}
