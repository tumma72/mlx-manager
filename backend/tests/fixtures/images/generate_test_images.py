"""Generate simple test images for E2E vision tests.

Run once: python -m tests.fixtures.images.generate_test_images
Or:       python tests/fixtures/images/generate_test_images.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent


def generate():
    """Generate all test images."""
    # Red square on white background
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([64, 64, 192, 192], fill="red")
    img.save(OUT / "red_square.png")

    # Blue circle on white background
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([64, 64, 192, 192], fill="blue")
    img.save(OUT / "blue_circle.png")

    # Text sample: "Hello MLX" on white background
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 100), "Hello MLX", fill="black", font=font)
    img.save(OUT / "text_sample.png")

    print(f"Generated test images in {OUT}")


if __name__ == "__main__":
    generate()
