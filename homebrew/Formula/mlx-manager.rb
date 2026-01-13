class MlxManager < Formula
  include Language::Python::Virtualenv

  desc "Web-based manager for MLX language models on Apple Silicon"
  homepage "https://github.com/tumma72/mlx-manager"
  url "https://files.pythonhosted.org/packages/source/m/mlx-manager/mlx_manager-1.0.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"

  depends_on "python@3.12"
  depends_on :macos
  depends_on arch: :arm64

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      To start MLX Manager:
        mlx-manager serve

      Then open http://localhost:8080 in your browser.

      To run as a background service:
        mlx-manager install-service

      To launch the menubar app:
        mlx-manager menubar
    EOS
  end

  service do
    run [opt_bin/"mlx-manager", "serve"]
    keep_alive true
    log_path var/"log/mlx-manager.log"
    error_log_path var/"log/mlx-manager.log"
  end

  test do
    assert_match "MLX Manager", shell_output("#{bin}/mlx-manager --help")
  end
end
