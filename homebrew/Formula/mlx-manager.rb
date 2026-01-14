class MlxManager < Formula
  include Language::Python::Virtualenv

  desc "Web-based manager for MLX language models on Apple Silicon"
  homepage "https://github.com/tumma72/mlx-manager"
  url "https://files.pythonhosted.org/packages/86/4a/e3962aa3c26990db27887905fed63dcc5aa6e1a9f4728bc4864cee697c57/mlx_manager-1.0.0.tar.gz"
  sha256 "590efcd138be891c47565c846ae3cc5e7381fe8f77501c89faf0bb5cc0dff727"
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

      For the menubar app:
        mlx-manager menubar

      To auto-start on login:
        mlx-manager install-service
    EOS
  end

  test do
    assert_match "MLX Manager", shell_output("#{bin}/mlx-manager --help")
  end
end
