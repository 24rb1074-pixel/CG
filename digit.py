from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("digits", exist_ok=True)

for n in range(10):
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("arial.ttf", 96)
    text = str(n)

    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    draw.text(
        ((128 - w) / 2, (128 - h) / 2 - 8),
        text,
        fill=(255, 255, 255, 255),
        font=font
    )

    img.save(f"digits/{n}.png")