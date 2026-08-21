"""
Batch convert all PDF files in a folder to JPG images.

Usage:
    python pdf_to_jpg.py --folder_dir /path/to/pdf_resumes --dpi 150

Output structure (same folder):
    resume_001.pdf  ->  resume_001_page1.jpg, resume_001_page2.jpg, ...
"""

import os
import argparse
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm


def pdf_to_jpg(pdf_path, output_dir, dpi=150, remove_images=False):
    """Convert a single PDF to JPG image(s). Returns list of saved paths."""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    saved_paths = []

    for page_num, page in enumerate(doc):
        if remove_images:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                for rect in page.get_image_rects(xref):
                    page.add_redact_annot(rect, fill=(1, 1, 1))
            page.apply_redactions()

        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if len(doc) == 1:
            out_name = f"{basename}.jpg"
        else:
            out_name = f"{basename}_page{page_num + 1}.jpg"

        out_path = os.path.join(output_dir, out_name)
        img.save(out_path, "JPEG", quality=95)
        saved_paths.append(out_path)

    doc.close()
    return saved_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_dir", type=str, required=True,
                        help="Directory containing PDF files.")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for rendering (default: 150).")
    parser.add_argument("--remove_images", action="store_true",
                        help="Remove all embedded images (profile photos, etc.) from the PDF before rendering.")
    parser.add_argument("--test_mode", action="store_true",
                        help="If set, process only the first 2 PDFs for testing.")
    args = parser.parse_args()

    output_dir = args.folder_dir.replace("pdf", "jpg")
    if args.remove_images:
        output_dir += "_no_images"
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = sorted([f for f in os.listdir(args.folder_dir) if f.lower().endswith(".pdf")])
    if args.test_mode:
        pdf_files = pdf_files[:2]
    print(f"Found {len(pdf_files)} PDF files in {args.folder_dir}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}\n")

    total_images = 0
    for fname in tqdm(pdf_files, desc="Converting PDFs"):
        pdf_path = os.path.join(args.folder_dir, fname)
        saved = pdf_to_jpg(pdf_path, output_dir, dpi=args.dpi, remove_images=args.remove_images)
        total_images += len(saved)

    print(f"\nDone. Converted {len(pdf_files)} PDFs -> {total_images} JPG images.")


if __name__ == "__main__":
    main()
