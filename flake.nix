{
  description = "bitmagnet remote classifier service";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
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
      # ~119 MB
      url = "https://huggingface.co/zopieux/e5-small-torrent-nsfw/resolve/main/backbone_int8.onnx";
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
      fasttext-model,
      e5-head-weights,
      e5-backbone,
      e5-tokenizer,
      e5-tokenizer-config,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        python = pkgs.python314;
        pythonPackages = pkgs.python314Packages;

        e5Model = pkgs.linkFarm "e5_nsfw" [
          {
            name = "head_weights.npz";
            path = e5-head-weights;
          }
          {
            name = "backbone_int8.onnx";
            path = e5-backbone;
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

        classifierPkg = pythonPackages.buildPythonApplication {
          pname = "bitmagnet-classify";
          version = "0.1.0";
          pyproject = true;

          src = ./.;

          build-system = [
            pythonPackages.setuptools
          ];

          dependencies = with pythonPackages; [
            fastapi
            fasttext
            httpx
            numpy
            onnxruntime
            transformers
            uvicorn
          ];

          makeWrapperArgs = [
            "--set"
            "FASTTEXT_MODEL_PATH"
            "${fasttext-model}"
            "--set"
            "NSFW_ONNX_PATH"
            "${e5Model}"
            "--set"
            "TRANSFORMERS_NO_ADVISORY_WARNINGS"
            "1"
          ];

          doCheck = false;

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
        packages.default = classifierPkg;

        devShells.default = pkgs.mkShell {
          inputsFrom = [ classifierPkg ];
          packages = [
            python
            pkgs.pyrefly
            pkgs.ruff
          ];
          FASTTEXT_MODEL_PATH = "${fasttext-model}";
          NSFW_ONNX_PATH = "${e5Model}";
          TRANSFORMERS_NO_ADVISORY_WARNINGS = "1";
        };

        devShells.cuda = cudaPkgs.mkShell {
          inputsFrom = [ classifierPkg ];
          packages = with cudaPythonPackages; [
            cudaPkgs.python314
            numpy
            torch
            transformers
            sentence-transformers
            scikit-learn
            onnx
            onnxruntime
            onnxscript
          ];
          FASTTEXT_MODEL_PATH = "${fasttext-model}";
          NSFW_ONNX_PATH = "${e5Model}";
          TRANSFORMERS_NO_ADVISORY_WARNINGS = "1";
        };
      }
    );
}
