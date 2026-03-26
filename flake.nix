{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };
  outputs = {
    nixpkgs,
    utils,
    rust-overlay,
    ...
  }:
    utils.lib.eachDefaultSystem (system: let
      # add/override rust packages in nixpkgs
      overlays = [(import rust-overlay)];
      # load nixpkgs with the current system and our overlays applied
      pkgs = import nixpkgs {
        inherit system overlays;
      };
      # stable rust packages; replace with whatever version you want
      rust-stable = pkgs.rust-bin.stable."1.90.0";
      # function to create a toolchain with whatever components you need
      # (clippy, rustfmt, etc.)
      mkToolchain = components:
        rust-stable.default.override {
          extensions = components;
          targets = ["wasm32-unknown-unknown"];
        };
    in {
      devShells.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          # rust stdlib + tools
          (mkToolchain ["rust-src" "clippy" "rustfmt" "rust-analyzer"])
          trunk
        ];
      };
    });
}
