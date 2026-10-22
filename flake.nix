{
  description = "bitmagnet remote classifier service";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    fasttext-model = {
      # ~0.9 MB
      url = "https://huggingface.co/julien-c/fasttext-language-id/resolve/main/lid.176.ftz";
      flake = false;
    };
    e5-head-weights = {
      # ~200 KB
      url = "https://huggingface.co/zopieux/e5-small-torrent-nsfw/resolve/main/head_weights.npz";
      flake = false;
    };
    e5-backbone = {
      # ~449 MB
      url = "https://huggingface.co/zopieux/e5-small-torrent-nsfw/resolve/main/backbone.onnx";
      flake = false;
    };
    e5-backbone-data = {
      # ~449 MB
      url = "https://huggingface.co/zopieux/e5-small-torrent-nsfw/resolve/main/backbone.onnx.data";
      flake = false;
    };
    e5-tokenizer = {
      # ~17 MB
      url = "https://huggingface.co/zopieux/e5-small-torrent-nsfw/resolve/main/tokenizer.json";
      flake = false;
    };
    e5-tokenizer-config = {
      # ~400 B
      url = "https://huggingface.co/zopieux/e5-small-torrent-nsfw/resolve/main/tokenizer_config.json";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      fasttext-model,
      e5-head-weights,
      e5-backbone,
      e5-backbone-data,
      e5-tokenizer,
      e5-tokenizer-config,
      ...
    }:
    let
      rustVersion = "1.99.0";
    in
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rustToolchain = pkgs.rust-bin.stable.${rustVersion}.minimal;

        devRustToolchain = pkgs.rust-bin.stable.${rustVersion}.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        e5Model = pkgs.linkFarm "e5_nsfw" [
          {
            name = "head_weights.npz";
            path = e5-head-weights;
          }
          {
            name = "backbone.onnx";
            path = e5-backbone;
          }
          {
            name = "backbone.onnx.data";
            path = e5-backbone-data;
          }
          {
            name = "tokenizer.json";
            path = e5-tokenizer;
          }
          {
            name = "tokenizer_config.json";
            path = e5-tokenizer-config;
          }
        ];

        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

        rustClassifierPkg = rustPlatform.buildRustPackage {
          pname = "bitmagnet-classify";
          version = "0.1.0";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
          };
          RUSTFLAGS = "--remap-path-prefix ${rustToolchain}=/rustc";
          nativeBuildInputs = [
            pkgs.pkg-config
            pkgs.makeWrapper
          ];
          buildInputs = [
            pkgs.onnxruntime
          ];
          postFixup = ''
            wrapProgram $out/bin/bitmagnet-classify \
              --set-default FASTTEXT_MODEL_PATH "${fasttext-model}" \
              --set-default NSFW_ONNX_PATH "${e5Model}" \
              --set-default ORT_DYLIB_PATH "${pkgs.onnxruntime}/lib/libonnxruntime.so"
          '';

          meta.mainProgram = "bitmagnet-classify";
        };

        cudaPkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        cudaPythonPackages = cudaPkgs.python314Packages;
      in
      {
        packages.default = rustClassifierPkg;

        devShells.default = pkgs.mkShell {
          packages = [
            devRustToolchain
            pkgs.pkg-config
            pkgs.onnxruntime
          ];
          FASTTEXT_MODEL_PATH = "${fasttext-model}";
          NSFW_ONNX_PATH = "${e5Model}";
          TRANSFORMERS_NO_ADVISORY_WARNINGS = "1";
          ORT_DYLIB_PATH = "${pkgs.onnxruntime}/lib/libonnxruntime.so";
        };

        devShells.cuda = cudaPkgs.mkShell {
          packages = [
            devRustToolchain
            cudaPkgs.pkg-config
            cudaPkgs.onnxruntime
          ] ++ (with cudaPythonPackages; [
            cudaPkgs.python314
            numpy
            torch
            transformers
            sentence-transformers
            scikit-learn
            onnx
            onnxruntime
            onnxscript
            cudaPkgs.pyrefly
            cudaPkgs.ruff
          ]);
          FASTTEXT_MODEL_PATH = "${fasttext-model}";
          NSFW_ONNX_PATH = "${e5Model}";
          TRANSFORMERS_NO_ADVISORY_WARNINGS = "1";
          ORT_DYLIB_PATH = "${cudaPkgs.onnxruntime}/lib/libonnxruntime.so";
          NSFW_DEVICE = "auto";
        };
      }
    );
}
