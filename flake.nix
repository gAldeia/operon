{
  description = "Operon development environment";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nur.url = "github:nix-community/NUR";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";

  outputs = { self, flake-utils, nixpkgs, nur }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ nur.overlay ];
          };
          repo = pkgs.nur.repos.foolnotion;

          # ceres has to be compiled from git due to a bug in tiny solver
          ceres-solver = pkgs.ceres-solver.overrideAttrs(old: rec {
              src = pkgs.fetchFromGitHub {
                repo   = "ceres-solver";
                owner  = "ceres-solver";
                rev    = "ce1537030b69cf9a4149d25fd7375dadef3e1f09";
                sha256 = "sha256-V3CGRilX5US8vx1cB2A5EQ7sqiIX82ARVNbeO3hkKk4=";
              };
          });
        in
        {
          devShell = pkgs.gcc11Stdenv.mkDerivation {
            name = "operon-env";
            hardeningDisable = [ "all" ];
            impureUseNativeOptimizations = true;
            nativeBuildInputs = with pkgs; [ bear cmake clang_13 clang-tools cppcheck include-what-you-use ];
            buildInputs = with pkgs; [
                # python environment for bindings and scripting
                (python39.override { stdenv = gcc11Stdenv; })
                (python39.withPackages (ps: with ps; [ pybind11 pytest pip pyperf colorama coloredlogs grip livereload joblib graphviz sphinx recommonmark sphinx_rtd_theme ]))
                # Project dependencies and utils for profiling and debugging
                ceres-solver
                cxxopts
                diff-so-fancy
                doctest
                eigen
                fmt
                gdb
                glog
                hotspot
                hyperfine
                jemalloc
                linuxPackages.perf
                mimalloc
                ninja
                openlibm
                pkg-config
                valgrind
                xxHash

                boost
                tbb

                # Some dependencies are provided by a NUR repo
                repo.aria-csv
                repo.autodiff
                repo.cmake-init
                repo.cmaketools
                repo.cpp-sort
                repo.fast_float
                repo.eli5
                repo.pmlb
                repo.pratt-parser
                repo.robin-hood-hashing
                repo.span-lite
                repo.taskflow
                repo.vectorclass
                repo.vstat
              ];

            shellHook = ''
              LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.gcc11Stdenv.cc.cc.lib ]};
              '';
          };
        }
      );
}
