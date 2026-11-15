"""Build ENS_492_PROGRESS_REPORT_II.docx from a copy of PR-I's docx (stdlib only).

Keeps PR-I's cover, styles, numbering, theme and section settings; edits the cover text,
replaces the table of contents with a fresh TOC field (Word updates it), and replaces the
body after the TOC with the blocks in content.py.

    python build_docx.py <pr1.docx> <out.docx> <figures_dir>
"""
from __future__ import annotations

import re
import struct
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).parent))
import content  # noqa: E402

TNR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
PAGE_W = 9072  # twips between margins (11906 - 2 * 1417)


def runs(text: str, *, base_rpr: str = "", size: int | None = None) -> str:
    """Inline markup: **bold**, *italic*, `code`."""
    out = []
    sz = f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>' if size else ""
    for tok in re.split(r"(\*\*.+?\*\*|`.+?`|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*]))", text):
        if not tok:
            continue
        rpr = TNR + base_rpr
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            tok, rpr = tok[2:-2], rpr + "<w:b/><w:bCs/>"
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            tok = tok[1:-1]
            rpr = ('<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'
                   + base_rpr)
            if size:
                sz2 = size - 2
                out.append(f'<w:r><w:rPr>{rpr}<w:sz w:val="{sz2}"/><w:szCs w:val="{sz2}"/>'
                           f'</w:rPr><w:t xml:space="preserve">{escape(tok)}</w:t></w:r>')
            else:
                out.append(f'<w:r><w:rPr>{rpr}<w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
                           f'<w:t xml:space="preserve">{escape(tok)}</w:t></w:r>')
            continue
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            tok, rpr = tok[1:-1], rpr + "<w:i/><w:iCs/>"
        out.append(f'<w:r><w:rPr>{rpr}{sz}</w:rPr><w:t xml:space="preserve">{escape(tok)}</w:t></w:r>')
    return "".join(out)


_bm = [900]


def h1(text):
    _bm[0] += 1
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
            f'<w:bookmarkStart w:id="{_bm[0]}" w:name="_TocPR2_{_bm[0]}"/>'
            f'<w:r><w:t>{escape(text)}</w:t></w:r><w:bookmarkEnd w:id="{_bm[0]}"/></w:p>')


def h2(text):
    _bm[0] += 1
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr>'
            f'<w:bookmarkStart w:id="{_bm[0]}" w:name="_TocPR2_{_bm[0]}"/>'
            f'{runs(text)}<w:bookmarkEnd w:id="{_bm[0]}"/></w:p>')


def h3(text):
    _bm[0] += 1
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading3"/></w:pPr>'
            f'<w:bookmarkStart w:id="{_bm[0]}" w:name="_TocPR2_{_bm[0]}"/>'
            f'{runs(text, base_rpr=H3RPR)}<w:bookmarkEnd w:id="{_bm[0]}"/></w:p>')


H3RPR = '<w:b/><w:bCs/><w:color w:val="000000"/><w:sz w:val="24"/><w:szCs w:val="24"/>'


def para(text, *, before=120, jc="both"):
    return (f'<w:p><w:pPr><w:spacing w:before="{before}" w:after="80"/><w:jc w:val="{jc}"/>'
            f'<w:rPr>{TNR}</w:rPr></w:pPr>{runs(text)}</w:p>')


def bullet(text):
    return (f'<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="10"/></w:numPr>'
            f'<w:spacing w:after="40"/><w:jc w:val="both"/><w:rPr>{TNR}</w:rPr></w:pPr>'
            f'{runs(text)}</w:p>')


def caption(text, *, center=False):
    m = re.match(r"((?:Figure|Table) \d+\. .*?\.)(\s.*|$)", text, flags=re.S)
    head, rest = (m.group(1), m.group(2)) if m else (text, "")
    jc = '<w:jc w:val="center"/>' if center else '<w:jc w:val="both"/>'
    return (f'<w:p><w:pPr><w:spacing w:before="60" w:after="200"/>{jc}'
            f'<w:rPr>{TNR}</w:rPr></w:pPr>'
            f'{runs(head, base_rpr="<w:b/><w:bCs/>", size=20)}'
            f'{runs(rest, size=20)}</w:p>')


def table_caption(text):
    m = re.match(r"(Table \d+\. .*?\.)(\s.*|$)", text, flags=re.S)
    head, rest = (m.group(1), m.group(2)) if m else (text, "")
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="60"/>'
            f'<w:rPr>{TNR}</w:rPr></w:pPr>'
            f'{runs(head, base_rpr="<w:b/><w:bCs/>", size=20)}'
            f'{runs(rest, size=20)}</w:p>')


def table(header, rows, widths, *, note=None, size=20):
    total = sum(widths)
    cols = [int(PAGE_W * w / total) for w in widths]
    border = ''.join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                     for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="{PAGE_W}" w:type="dxa"/><w:tblBorders>{border}'
           f'</w:tblBorders><w:tblLayout w:type="fixed"/>'
           f'<w:tblCellMar><w:left w:w="70" w:type="dxa"/><w:right w:w="70" w:type="dxa"/>'
           f'</w:tblCellMar><w:tblLook w:val="0000"/></w:tblPr><w:tblGrid>'
           + "".join(f'<w:gridCol w:w="{c}"/>' for c in cols) + "</w:tblGrid>"]

    def row(cells, head=False):
        trpr = '<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>' if head else '<w:trPr><w:cantSplit/></w:trPr>'
        r = [f"<w:tr>{trpr}"]
        for c, w in zip(cells, cols):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if head else ""
            paras_ = str(c).split("\n")
            ps = "".join(
                f'<w:p><w:pPr><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>'
                f'<w:rPr>{TNR}</w:rPr></w:pPr>'
                f'{runs(t, base_rpr=("<w:b/><w:bCs/>" if head else ""), size=size)}</w:p>' for t in paras_)
            r.append(f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shade}</w:tcPr>{ps}</w:tc>')
        r.append("</w:tr>")
        return "".join(r)

    out.append(row(header, head=True))
    out.extend(row(r) for r in rows)
    out.append("</w:tbl>")
    if note:
        out.append(f'<w:p><w:pPr><w:spacing w:before="40" w:after="160"/><w:jc w:val="both"/>'
                   f'<w:rPr>{TNR}</w:rPr></w:pPr>{runs(note, size=18)}</w:p>')
    else:
        out.append('<w:p><w:pPr><w:spacing w:before="0" w:after="80"/></w:pPr></w:p>')
    return "".join(out)


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


_pic = [5000]


def figure(rid, path, width_in):
    w, h = png_size(path)
    cx = int(width_in * 914400)
    cy = int(cx * h / w)
    _pic[0] += 1
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="0"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{_pic[0]}" name="Picture {_pic[0]}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="0" name="{Path(path).name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def refentry(text):
    return (f'<w:p><w:pPr><w:pStyle w:val="EndNoteBibliography"/><w:spacing w:after="60"/>'
            f'<w:ind w:left="720" w:hanging="720"/></w:pPr>{runs(text)}</w:p>')


def pagebreak():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


TOC = ('<w:p><w:pPr><w:pStyle w:val="TOCHeading"/></w:pPr><w:r><w:rPr>' + TNR +
       '<w:b/><w:bCs/></w:rPr><w:t>Table of Contents</w:t></w:r></w:p>'
       '<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
       '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'
       '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
       '<w:r><w:rPr>' + TNR + '<w:i/></w:rPr><w:t>[Right-click here and choose Update Field to build the table of contents.]</w:t></w:r>'
       '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>' + pagebreak())


def main(src, dst, figdir):
    figdir = Path(figdir)
    zin = zipfile.ZipFile(src)
    doc = zin.read("word/document.xml").decode("utf8")
    rels = zin.read("word/_rels/document.xml.rels").decode("utf8")
    settings = zin.read("word/settings.xml").decode("utf8")

    # --- cover text edits (exact strings from PR-I)
    a, b = "ENS 491 – Graduation Project (Design)", "ENS 492 – Graduation Project (Implementation)"
    assert doc.count(a) == 1, a
    doc = doc.replace(a, b)
    # Date is split over runs: "May" | " 24" | "th" | ...
    i = doc.find(">Date<")
    assert i > 0
    seg = doc[i:i + 3000]
    assert seg.count("<w:t>May</w:t>") == 1 and seg.count("> 24</w:t>") == 1, seg[:600]
    seg = seg.replace("<w:t>May</w:t>", "<w:t>November</w:t>").replace("> 24</w:t>", "> 15</w:t>")
    doc = doc[:i] + seg + doc[i + 3000:]
    # "Progress Report I" may be split over runs; replace the run text that ends the title
    n = len(re.findall(r">Progress Report I<", doc))
    if n == 1:
        doc = doc.replace(">Progress Report I<", ">Progress Report II<")
    else:
        m = re.search(r"Progress Report I(?!I)", doc)
        assert m, "cover title not found"
        doc = doc[:m.end()] + "I" + doc[m.end():]

    # --- split: cover | TOC sdt | body | sectPr
    sdt_start = doc.find("<w:sdt>")
    sdt_end = doc.find("</w:sdt>") + len("</w:sdt>")
    assert 0 < sdt_start < sdt_end
    sect = doc.rfind("<w:sectPr")
    head = doc[:sdt_start]
    tail = doc[sect:]

    # --- figures
    rel_ids = {}
    media = {}
    for i, (key, fname) in enumerate(content.FIGURES.items()):
        rid = f"rIdPR2fig{i + 1}"
        rel_ids[key] = rid
        media[f"word/media/pr2_{fname}"] = (figdir / fname).read_bytes()
        rels = rels.replace(
            "</Relationships>",
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/pr2_{fname}"/></Relationships>')

    body = []
    for blk in content.BLOCKS:
        kind = blk[0]
        if kind == "h1":
            body.append(h1(blk[1]))
        elif kind == "h2":
            body.append(h2(blk[1]))
        elif kind == "h3":
            body.append(h3(blk[1]))
        elif kind == "p":
            body.append(para(blk[1]))
        elif kind == "b":
            body.append(bullet(blk[1]))
        elif kind == "fig":
            _, key, width, cap = blk
            body.append(figure(rel_ids[key], figdir / content.FIGURES[key], width))
            body.append(caption(cap))
        elif kind == "tab":
            _, cap, header, rows, widths, note = blk
            body.append(table_caption(cap))
            body.append(table(header, rows, widths, note=note))
        elif kind == "ref":
            body.append(refentry(blk[1]))
        elif kind == "pb":
            body.append(pagebreak())
        else:
            raise ValueError(kind)

    new_doc = head + TOC + "".join(body) + tail

    # PR-I was formatted with EndNote; its document variables make the EndNote add-in look for
    # PR-I's library on every open. The new body has no EndNote fields, so drop them.
    settings = re.sub(r'<w:docVar w:name="EN\.[^"]*" w:val="[^"]*"/>', "", settings)
    settings = re.sub(r"<w:docVars>\s*</w:docVars>|<w:docVars/>", "", settings)
    if "<w:updateFields" not in settings:
        settings = re.sub(r"(<w:settings[^>]*>)", r'\1<w:updateFields w:val="true"/>', settings, count=1)

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename.startswith("word/media/image"):
                continue  # PR-I's figures are not used
            if item.filename == "word/document.xml":
                zout.writestr(item, new_doc.encode("utf8"))
            elif item.filename == "word/_rels/document.xml.rels":
                rels2 = re.sub(r'<Relationship Id="[^"]+" Type="[^"]+/image" Target="media/image\d+\.png"/>', "", rels)
                zout.writestr(item, rels2.encode("utf8"))
            elif item.filename == "word/settings.xml":
                zout.writestr(item, settings.encode("utf8"))
            else:
                zout.writestr(item, zin.read(item.filename))
        for name, data in media.items():
            zout.writestr(name, data)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:4])
