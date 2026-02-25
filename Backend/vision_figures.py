"""
Figure/diagram extraction and description using Groq vision.
"""

import os
import base64
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

EXT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


def _get_groq_client():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)


def describe_image_with_groq(
    image_bytes: bytes,
    ext: str = "png",
    prompt_override: Optional[str] = None,
) -> str:
    client = _get_groq_client()
    mime = EXT_TO_MIME.get(ext.lower(), "image/png")
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    url = f"data:{mime};base64,{b64}"
    prompt = prompt_override or (
        "Analyze this image in detail. If it is a diagram, chart, or figure: "
        "describe what it shows, the main elements, labels, and any data or trends. "
        "If it contains text, transcribe it. If it is a table, describe structure and content. "
        "Output in clear Markdown. Be concise but complete."
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }],
            max_tokens=2048,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"Groq vision failed for image: {e}")
        return ""


def extract_figures_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF (fitz) not available; skipping figure extraction")
        return []
    if not file_path.lower().endswith(".pdf"):
        return []
    figures = []
    try:
        doc = fitz.open(file_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_no = page_idx + 1
            for fig_idx, img_ref in enumerate(page.get_images()):
                xref = img_ref[0]
                try:
                    img = doc.extract_image(xref)
                    image_bytes = img.get("image")
                    ext = (img.get("ext") or "png").lower()
                    if not image_bytes or len(image_bytes) < 100:
                        continue
                    figures.append({
                        "page_no": page_no,
                        "figure_index": fig_idx + 1,
                        "image_bytes": image_bytes,
                        "ext": ext,
                    })
                except Exception as e:
                    logger.debug(f"Skip image xref {xref} on page {page_no}: {e}")
        doc.close()
    except Exception as e:
        logger.error(f"PDF figure extraction failed: {e}", exc_info=True)
        return []
    return figures


def process_figures_for_document(
    file_path: str, user_id: str, doc_id: int, filename: str,
) -> List[Dict[str, Any]]:
    figures = extract_figures_from_pdf(file_path)
    if not figures:
        return []
    results = []
    for fig in figures:
        desc = describe_image_with_groq(fig["image_bytes"], ext=fig["ext"])
        if not desc:
            continue
        page_no, idx = fig["page_no"], fig["figure_index"]
        text_for_chunk = f"Figure on page {page_no} (diagram {idx}):\n{desc}"
        results.append({
            "page_no": page_no,
            "figure_index": idx,
            "description": desc,
            "text_for_chunk": text_for_chunk,
        })
    return results