class MlxManager < Formula
  include Language::Python::Virtualenv

  desc "Web-based MLX model manager for Apple Silicon Macs"
  homepage "https://github.com/tumma72/mlx-manager"
  url "https://github.com/tumma72/mlx-manager/archive/refs/tags/v1.2.7.tar.gz"
  sha256 "7b2f69369c5bbce737171766a85b146523323bb4d47a7b681b95f70d6afeb204"
  license "MIT"
  version "1.2.7"

  depends_on "python@3.12"
  depends_on "uv"
  depends_on :macos
  depends_on arch: :arm64

  def install
    # Create virtualenv - package install happens in post_install to avoid dylib fixup
    virtualenv_create(libexec, "python3.12")

    # Create a wrapper script so it gets linked during install
    # (post_install runs after linking, so bin.install_symlink there won't be on PATH)
    (bin/"mlx-manager").write <<~SH
      #!/bin/bash
      exec "#{libexec}/bin/mlx-manager" "$@"
    SH
    (bin/"mlx-manager").chmod 0755
  end

  def post_install
    # Write dependency overrides to resolve upstream version conflicts
    # (mlx-audio/mlx-lm pin transformers==5.0.0rc3 but mlx-vlm requires >=5.1.0)
    overrides = libexec/"overrides.txt"
    overrides.write <<~EOS
      mlx-lm>=0.30.5
      transformers>=5.0.0
    EOS

    # Install from PyPI using uv for override support
    # --prerelease=allow needed because mlx-audio pins transformers==5.0.0rc3
    system "uv", "pip", "install",
           "--python", libexec/"bin/python",
           "--no-cache-dir",
           "--prerelease=allow",
           "--overrides", overrides,
           "mlx-manager==#{version}"

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
