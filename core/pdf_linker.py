import re
from typing import Dict, Any, List, Optional, Tuple
import fitz  # PyMuPDF


# Regex Patterns for links
URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+(?:/[^\s]*)?",
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
    re.IGNORECASE
)

PAGE_REF_PATTERN = re.compile(
    r"\b(?:see|refer\s+to|page|p\.)\s*(?:page\s*)?(\d+)\b",
    re.IGNORECASE
)


def find_and_create_links_in_pdf(
    doc: fitz.Document,
    link_urls: bool = True,
    link_emails: bool = True,
    link_page_refs: bool = True,
    custom_links: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Detects URLs, email addresses, internal page cross-references, and custom keywords,
    and inserts clickable PDF Link Annotations at their visual coordinates.

    Returns:
        Summary dict containing counts and list of all created link metadata.
    """
    total_pages = len(doc)
    created_links: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_text = page.get_text("text")

        # 1. URL Linking
        if link_urls:
            for match in URL_PATTERN.finditer(page_text):
                raw_url = match.group(0).rstrip(".,;:)")
                target_url = raw_url if raw_url.startswith(("http://", "https://")) else f"https://{raw_url}"
                rects = page.search_for(raw_url)
                for rect in rects:
                    link_dict = {
                        "kind": fitz.LINK_URI,
                        "from": rect,
                        "uri": target_url
                    }
                    page.insert_link(link_dict)
                    created_links.append({
                        "page": page_num,
                        "type": "URL",
                        "text": raw_url,
                        "target": target_url,
                        "bbox": (rect.x0, rect.y0, rect.x1, rect.y1)
                    })

        # 2. Email Linking
        if link_emails:
            for match in EMAIL_PATTERN.finditer(page_text):
                email = match.group(0).rstrip(".,;:)")
                target_mailto = f"mailto:{email}"
                rects = page.search_for(email)
                for rect in rects:
                    link_dict = {
                        "kind": fitz.LINK_URI,
                        "from": rect,
                        "uri": target_mailto
                    }
                    page.insert_link(link_dict)
                    created_links.append({
                        "page": page_num,
                        "type": "Email",
                        "text": email,
                        "target": target_mailto,
                        "bbox": (rect.x0, rect.y0, rect.x1, rect.y1)
                    })

        # 3. Internal Page Reference Linking ("See page X")
        if link_page_refs:
            for match in PAGE_REF_PATTERN.finditer(page_text):
                full_phrase = match.group(0)
                target_page_str = match.group(1)
                try:
                    target_page_int = int(target_page_str)
                    # Convert 1-indexed human page to 0-indexed PyMuPDF target
                    if 1 <= target_page_int <= total_pages:
                        rects = page.search_for(full_phrase)
                        for rect in rects:
                            link_dict = {
                                "kind": fitz.LINK_GOTO,
                                "from": rect,
                                "page": target_page_int - 1
                            }
                            page.insert_link(link_dict)
                            created_links.append({
                                "page": page_num,
                                "type": "Internal Page Jump",
                                "text": full_phrase,
                                "target": f"Page {target_page_int}",
                                "bbox": (rect.x0, rect.y0, rect.x1, rect.y1)
                            })
                except ValueError:
                    continue

        # 4. Custom Keyword Links
        if custom_links:
            for keyword, target_url in custom_links.items():
                kw_clean = keyword.strip()
                if not kw_clean or not target_url.strip():
                    continue
                rects = page.search_for(kw_clean)
                for rect in rects:
                    link_dict = {
                        "kind": fitz.LINK_URI,
                        "from": rect,
                        "uri": target_url.strip()
                    }
                    page.insert_link(link_dict)
                    created_links.append({
                        "page": page_num,
                        "type": "Custom Link",
                        "text": kw_clean,
                        "target": target_url.strip(),
                        "bbox": (rect.x0, rect.y0, rect.x1, rect.y1)
                    })

    return {
        "total_links": len(created_links),
        "links": created_links
    }
