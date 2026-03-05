class MlxManager < Formula
  include Language::Python::Virtualenv

  desc "Web-based MLX model manager for Apple Silicon Macs"
  homepage "https://github.com/tumma72/mlx-manager"
  url "https://github.com/tumma72/mlx-manager/archive/refs/tags/v1.2.1.tar.gz"
  sha256 "c27cd3a709ae996bbc7f710781903c71e181fa537412f68d6c018bc1e6c315c4"
  license "MIT"
  version "1.2.1"

  depends_on "python@3.12"
  depends_on "uv"
  depends_on :macos
  depends_on arch: :arm64

  def install
    # Create virtualenv - package install happens in post_install to avoid dylib fixup
    virtualenv_create(libexec, "python3.12")

    # Create the bin symlink directory
    bin.mkpath
  end

  def post_install
    # Write dependency overrides to resolve upstream version conflicts
    # (mlx-audio/mlx-lm pin transformers==5.0.0rc3 but mlx-vlm requires >=5.1.0)
    overrides = libexec/"overrides.txt"
    overrides.write <<~EOS
      mlx-lm>=0.30.5
      transformers>=5.0.0rc3
    EOS

    # Install from PyPI using uv for override support
    system "uv", "pip", "install",
           "--python", libexec/"bin/python",
           "--no-cache-dir",
           "--overrides", overrides,
           "mlx-manager==#{version}"

    # Link the binary
    bin.install_symlink libexec/"bin/mlx-manager"
  end

  def caveats
    <<~EOS
      To start the MLX Manager server:
        mlx-manager serve

      To run as a background service:
        brew services start mlx-manager

      To launch the menubar app:
        mlx-manager menubar

      The web interface will be available at:
        http://localhost:10242
    EOS
  end

  service do
    run [opt_bin/"mlx-manager", "serve", "--no-open"]
    keep_alive true
    log_path var/"log/mlx-manager.log"
    error_log_path var/"log/mlx-manager.log"
    working_dir HOMEBREW_PREFIX
  end

  test do
    assert_match "serve", shell_output("#{bin}/mlx-manager --help")
  end
end
