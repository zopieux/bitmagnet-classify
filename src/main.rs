use anyhow::{Context, Result};
use clap::Parser;
use std::collections::HashSet;
use std::os::unix::io::FromRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use bitmagnet_classify::classifier::Classifier;
use bitmagnet_classify::langid::LanguageDetector;
use bitmagnet_classify::nsfw::NsfwModel;
use bitmagnet_classify::server::create_router;

#[derive(Parser, Debug)]
#[command(
    name = "bitmagnet-classify",
    about = "bitmagnet classification service"
)]
struct Args {
    #[arg(long, default_value = "0.0.0.0", env = "HOST")]
    host: String,

    #[arg(long, default_value_t = 8080, env = "PORT")]
    port: u16,

    #[arg(long, env = "FD")]
    fd: Option<i32>,

    #[arg(long, env = "UDS")]
    uds: Option<PathBuf>,

    #[arg(long, value_delimiter = ' ', num_args = 0.., env = "EARLY_LANG")]
    early_lang: Vec<String>,

    #[arg(long, env = "NSFW_ONNX_PATH")]
    model_dir: Option<PathBuf>,

    #[arg(long, env = "FASTTEXT_MODEL_PATH")]
    fasttext_model_path: Option<PathBuf>,

    #[arg(long, env = "ORT_DYLIB_PATH")]
    ort_dylib_path: Option<String>,

    #[arg(long, default_value_t = 1)]
    threads: usize,
}

fn get_systemd_listen_fd() -> Option<i32> {
    let listen_pid = std::env::var("LISTEN_PID").ok()?;
    let listen_fds = std::env::var("LISTEN_FDS").ok()?;
    let pid: u32 = listen_pid.parse().ok()?;
    let fds: i32 = listen_fds.parse().ok()?;

    if pid == std::process::id() && fds >= 1 {
        Some(3) // SD_LISTEN_FDS_START
    } else {
        None
    }
}

fn get_socket_family(fd: i32) -> std::io::Result<i32> {
    let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let res = unsafe {
        libc::getsockname(
            fd,
            &mut storage as *mut _ as *mut libc::sockaddr,
            &mut len,
        )
    };
    if res != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(storage.ss_family as i32)
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

    let nsfw_model = if let Some(ref dir) = args.model_dir {
        if !dir.exists() {
            anyhow::bail!("NSFW model directory does not exist: {:?}", dir);
        }
        info!("NSFW model: enabled ({:?})", dir);
        Some(
            NsfwModel::load(dir, args.ort_dylib_path.as_deref(), args.threads)
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

    let mut default_early_langs: HashSet<String> = HashSet::new();
    for item in args.early_lang {
        for part in item.replace(',', " ").split_whitespace() {
            let code = part.trim().to_lowercase();
            if !code.is_empty() {
                default_early_langs.insert(code);
            }
        }
    }
    if !default_early_langs.is_empty() {
        info!("Default early-return languages: {:?}", default_early_langs);
    }

    let classifier = Arc::new(Classifier::new(
        lang_detector,
        nsfw_model,
        default_early_langs,
    ));

    let app = create_router(classifier);

    let systemd_fd = get_systemd_listen_fd();
    let fd = args.fd.or(systemd_fd);

    if let Some(fd_num) = fd {
        let family = get_socket_family(fd_num)
            .with_context(|| format!("Failed to inspect socket family of fd {fd_num}"))?;
        if family == libc::AF_UNIX as i32 {
            info!("Listening on Unix domain socket fd {fd_num} (socket activation)");
            let std_listener = unsafe { std::os::unix::net::UnixListener::from_raw_fd(fd_num) };
            std_listener.set_nonblocking(true)?;
            let listener = tokio::net::UnixListener::from_std(std_listener)?;
            axum::serve(listener, app).await?;
        } else if family == libc::AF_INET as i32 || family == libc::AF_INET6 as i32 {
            info!("Listening on TCP socket fd {fd_num} (socket activation)");
            let std_listener = unsafe { std::net::TcpListener::from_raw_fd(fd_num) };
            std_listener.set_nonblocking(true)?;
            let listener = tokio::net::TcpListener::from_std(std_listener)?;
            axum::serve(listener, app).await?;
        } else {
            anyhow::bail!("Unsupported socket family {family} on fd {fd_num}");
        }
    } else if let Some(uds_path) = args.uds {
        info!("Listening on UNIX domain socket {:?}", uds_path);
        if uds_path.exists() {
            let _ = std::fs::remove_file(&uds_path);
        }
        let listener = tokio::net::UnixListener::bind(&uds_path)?;
        axum::serve(listener, app).await?;
    } else {
        let addr = format!("{}:{}", args.host, args.port);
        info!("Listening on http://{}", addr);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
    }

    Ok(())
}
