class MlxManager < Formula
  include Language::Python::Virtualenv

  desc "Web-based MLX model manager for Apple Silicon Macs"
  homepage "https://github.com/tumma72/mlx-manager"
  url "https://github.com/tumma72/mlx-manager/archive/refs/tags/v1.1.0.tar.gz"
  sha256 "7b7d562b7d53df2798711d0c2981a636198032df91c0290e5310f5cc7b67f075"
  license "MIT"
  version "1.1.0"

  depends_on "python@3.12"
  depends_on :macos
  depends_on arch: :arm64

  def install
    # Create virtualenv - pip install happens in post_install to avoid dylib fixup
    virtualenv_create(libexec, "python3.12")

    # Create the bin symlink directory
    bin.mkpath
  end

  def post_install
    # Install from PyPI in post_install to bypass Homebrew's dylib relocation
    system libexec/"bin/python", "-m", "pip", "install", "--upgrade", "pip"
    system libexec/"bin/pip", "install", "--no-cache-dir", "mlx-manager==#{version}"

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
        http://localhost:8080
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
