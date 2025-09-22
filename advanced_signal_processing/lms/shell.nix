{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3Packages.numpy
    python3Packages.matplotlib
    python3Packages.sounddevice
    python3Packages.tqdm
    python3Packages.scipy
  ];
}
