from typing import Iterable, List


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_lines(lines: Iterable[str], width: int = 96) -> List[str]:
    out: List[str] = []
    for line in lines:
        cur = line.strip()
        if not cur:
            out.append("")
            continue
        while len(cur) > width:
            split_at = cur.rfind(" ", 0, width)
            if split_at <= 0:
                split_at = width
            out.append(cur[:split_at].strip())
            cur = cur[split_at:].strip()
        out.append(cur)
    return out


def build_basic_pdf(lines: Iterable[str]) -> bytes:
    """Generate a minimal valid one-page PDF without external dependencies."""
    wrapped = _wrap_lines(lines)

    content = ["BT", "/F1 10 Tf", "40 800 Td"]
    for idx, line in enumerate(wrapped[:52]):
        safe = _escape_pdf_text(line)
        content.append(f"({safe}) Tj")
        if idx < 51:
            content.append("0 -14 Td")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1", "replace")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream\nendobj\n",
    ]

    header = b"%PDF-1.4\n"
    body = bytearray(header)
    offsets = [0]
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    xref_pos = len(body)
    body.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        body.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    body.extend(
        (
            "trailer\n"
            f"<< /Size {len(offsets)} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_pos}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    return bytes(body)
