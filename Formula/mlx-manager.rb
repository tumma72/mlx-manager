class MlxManager < Formula
  include Language::Python::Virtualenv

  desc "Web-based MLX model manager for Apple Silicon Macs"
  homepage "https://github.com/tumma72/mlx-manager"
  url "https://files.pythonhosted.org/packages/source/m/mlx-manager/mlx_manager-1.0.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"

  depends_on "python@3.12"
  depends_on :macos

  def install
    virtualenv_install_with_resources
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
    assert_match "MLX Manager", shell_output("#{bin}/mlx-manager version")
  end
end
