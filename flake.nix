{
  description = "RT-3DGS POC: COLMAP + gsplat";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # CUDA toolkit
      };
      # torch only requires nvcc's CUDA *major* to match its own (cu128 wheel -> any 12.x)
      cuda = pkgs.cudaPackages_12;

      mkPocShell = { torchExtra, packages ? [ ], extraHook ? "" }:
        pkgs.mkShell {
          packages = with pkgs; [ python312 uv colmap ffmpeg ] ++ packages;
          shellHook = ''
            # pip wheels (torch, numpy, ...) expect FHS libs
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}
            export UV_PYTHON=${pkgs.python312}/bin/python3
            export UV_PYTHON_DOWNLOADS=never
            ${extraHook}
            uv sync --extra ${torchExtra}
            source .venv/bin/activate
          '';
        };
    in
    {
      devShells.${system} = {
        # NVIDIA machines
        default = mkPocShell {
          torchExtra = "cu128";
          packages = [ cuda.cudatoolkit ];
          extraHook = ''
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH  # libcuda.so
            # gsplat JIT-compiles its kernels on first use
            export CUDA_HOME=${cuda.cudatoolkit}
            export CC=${cuda.backendStdenv.cc}/bin/gcc   # nvcc-compatible host compiler
            export CXX=${cuda.backendStdenv.cc}/bin/g++
            export PATH=${pkgs.ninja}/bin:$PATH
          '';
        };

        # Laptop (AMD iGPU): CPU torch, no gsplat
        cpu = mkPocShell { torchExtra = "cpu"; };
      };
    };
}
