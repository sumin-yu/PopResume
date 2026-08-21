import os
import argparse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from PIL import Image

# ======================
# Args
# ======================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Generate only a small number of PDFs per folder for testing"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=10,
        help="Number of samples per folder when test_mode is enabled"
    )
    return parser.parse_args()


# ======================
# Config
# ======================
# Run this from the repository root (paths below are relative to it).
BASE_DIR = "Resume/job-distribution"

# Profile photos, one file per resume id (see the README on Face-MoGLE generation).
# The paper's run used the full set; pointing IMG_DIR at a directory whose name contains
# "new" writes to *_resumes_pdf_new instead, which is only useful for partial re-renders.
IMG_DIR = "./resume_images"

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 2 * cm
IMAGE_SIZE = 4 * cm


# ======================
# Helpers
# ======================
def find_image(resume_id):
    for ext in [".jpg", ".png", ".jpeg"]:
        path = os.path.join(IMG_DIR, str(resume_id) + ext)
        # print(f"Path: {path}")
        if os.path.exists(path):
            return path
    return None


from reportlab.lib.colors import HexColor

def txt_to_pdf(txt_path, out_pdf_path):
    base = os.path.splitext(os.path.basename(txt_path))[0]
    resume_id = int(base.split("_")[1])

    img_path = find_image(resume_id)

    c = canvas.Canvas(out_pdf_path, pagesize=A4)
    y = PAGE_HEIGHT - MARGIN

    # -----------------
    # Profile image
    # -----------------
    if img_path:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        ratio = h / w

        draw_w = IMAGE_SIZE
        draw_h = IMAGE_SIZE * ratio

        c.drawImage(
            img_path,
            PAGE_WIDTH - MARGIN - draw_w,
            y - draw_h,
            width=draw_w,
            height=draw_h,
            mask="auto",
        )
    else:
        return "Fail" # Skip if no image found

    text = c.beginText(MARGIN, y)
    text.setLeading(14)

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [l.rstrip() for l in f]

    # -----------------
    # Name (1st line)
    # -----------------
    name = lines[0]
    text.setFont("Helvetica-Bold", 16)
    text.textLine(name)
    text.moveCursor(0, 6)

    # -----------------
    # Contact (2nd line)
    # -----------------
    contact = lines[1]
    text.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#555555"))
    text.textLine(contact)
    c.setFillColor(HexColor("#000000"))
    text.moveCursor(0, 14)

    # -----------------
    # Body
    # -----------------
    for line in lines[2:]:
        if not line:
            text.moveCursor(0, 8)
            continue

        # SECTION HEADER
        if line.isupper():
            text.moveCursor(0, 12)
            text.setFont("Helvetica-Bold", 12)
            text.textLine(line)
            text.moveCursor(0, 6)
            continue

        # Bullet
        if line.startswith("- "):
            text.setFont("Helvetica", 10)
            text.textLine("   • " + line[2:])
            continue

        # Job / Education line
        text.setFont("Helvetica", 10)
        text.textLine(line)

        if text.getY() < MARGIN:
            c.drawText(text)
            c.showPage()
            text = c.beginText(MARGIN, PAGE_HEIGHT - MARGIN)
            text.setLeading(14)

    c.drawText(text)
    c.save()
    return "Success"

# ======================
# Main
# ======================
def main():
    args = parse_args()

    for folder in sorted(os.listdir(BASE_DIR)):
        if not folder.endswith("_resumes_txt"):
            continue

        txt_dir = os.path.join(BASE_DIR, folder)
        if "new" in IMG_DIR :
            pdf_dir = os.path.join(
                BASE_DIR,
                folder.replace("_resumes_txt", "_resumes_pdf_new")
            )
        else:
            pdf_dir = os.path.join(
                BASE_DIR,
                folder.replace("_resumes_txt", "_resumes_pdf")
            )
        os.makedirs(pdf_dir, exist_ok=True)

        txt_files = sorted(
            f for f in os.listdir(txt_dir) if f.endswith(".txt") and "with" not in f
        )

        if args.test_mode:
            txt_files = txt_files[:args.max_test_samples]

        for fname in txt_files:
            txt_path = os.path.join(txt_dir, fname)
            out_pdf = os.path.splitext(fname)[0] + "_img.pdf"
            out_pdf_path = os.path.join(pdf_dir, out_pdf)

            result = txt_to_pdf(txt_path, out_pdf_path)

        print(
            f"✅ {folder}: generated {len(txt_files)} PDFs"
            + (" (TEST MODE)" if args.test_mode else "")
        )


if __name__ == "__main__":
    main()