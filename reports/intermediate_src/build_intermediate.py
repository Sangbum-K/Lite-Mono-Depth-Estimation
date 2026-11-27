"""Build the intermediate report draft (.docx) on top of the proposal .docx template.

Keeps the proposal's cover page, fonts, paragraph/table formatting and footer; replaces the body.
Markup inside text: **bold**, ==yellow highlight== (places the author must fill in / confirm).
"""
import os
import re
import shutil
import sys
import zipfile
from xml.sax.saxutils import escape

from PIL import Image

REPO = "/home/user/sangbum/Lite-Mono-Depth-Estimation"
SRC = os.path.join(REPO, "reports/HW2_Proposal_Report.docx")
OUT = os.path.join(REPO, "reports/HW2_Intermediate_Report_draft.docx")
FIG = os.path.join(REPO, "reports/figures/intermediate")
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inter_docx")  # temp unzip dir (gitignored)

FONT = '<w:rFonts w:ascii="맑은 고딕" w:eastAsia="맑은 고딕" w:hAnsi="맑은 고딕"/>'
FULL_W = 9638   # twips, page width minus margins
COL_W = 4606    # twips, one column (as in the proposal tables)
EMU_PER_TWIP = 635

# ---------------------------------------------------------------- primitives

def runs(text, size=18, bold=False):
    out = []
    for part in re.split(r"(\*\*.+?\*\*|==.+?==)", text):
        if not part:
            continue
        b, hl = bold, False
        if part.startswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("=="):
            part, hl = part[2:-2], True
        rpr = FONT + ("<w:b/>" if b else "") + ('<w:highlight w:val="yellow"/>' if hl else "")
        if size:
            rpr += '<w:sz w:val="{0}"/><w:szCs w:val="{0}"/>'.format(size)
        rpr += '<w:lang w:eastAsia="ko-KR"/>'
        out.append('<w:r><w:rPr>{}</w:rPr><w:t xml:space="preserve">{}</w:t></w:r>'.format(rpr, escape(part)))
    return "".join(out)


def para(ppr, body):
    return '<w:p><w:pPr>{}<w:rPr>{}<w:lang w:eastAsia="ko-KR"/></w:rPr></w:pPr>{}</w:p>'.format(ppr, FONT, body)


AUTO = '<w:autoSpaceDE w:val="0"/><w:autoSpaceDN w:val="0"/>'


def H1(t):
    return para(AUTO + '<w:keepNext/><w:spacing w:before="200" w:after="80" w:line="240" w:lineRule="auto"/><w:outlineLvl w:val="0"/>',
                runs(t, size=None, bold=True))


def H2(t):
    return para(AUTO + '<w:keepNext/><w:spacing w:before="100" w:after="40" w:line="240" w:lineRule="auto"/><w:outlineLvl w:val="1"/>',
                runs(t, size=20, bold=True))


def P(t, indent=True):
    ind = '<w:ind w:firstLine="180"/>' if indent else ""
    return para(AUTO + '<w:spacing w:after="60" w:line="240" w:lineRule="auto"/>' + ind + '<w:jc w:val="both"/>', runs(t))


def CAP(t, before=60, after=40, keep_next=True):
    kn = "<w:keepNext/>" if keep_next else ""
    return para(AUTO + kn + '<w:spacing w:before="{}" w:after="{}" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>'.format(before, after),
                runs(t, size=16))


def FIGCAP(t):
    # figure captions sit below the figure, left-justified like a paragraph but smaller
    return para(AUTO + '<w:spacing w:before="40" w:after="160" w:line="240" w:lineRule="auto"/><w:jc w:val="both"/>', runs(t, size=16))


def REF(t):
    return para(AUTO + '<w:spacing w:after="30" w:line="240" w:lineRule="auto"/><w:ind w:left="340" w:hanging="340"/>', runs(t, size=15))


def SPACER(h=120):
    return para('<w:spacing w:line="{}" w:lineRule="exact"/>'.format(h), "")


def SECT(cols, footer=False, start=None):
    """Paragraph that ENDS a section. cols=1 or 2. All body sections are continuous."""
    s = '<w:sectPr>'
    if footer:
        s += '<w:footerReference w:type="default" r:id="rId9"/>'
    s += '<w:type w:val="continuous"/>'
    s += '<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="720" w:footer="567" w:gutter="0"/>'
    if start is not None:
        s += '<w:pgNumType w:start="{}"/>'.format(start)
    s += '<w:cols w:num="2" w:space="425"/>' if cols == 2 else '<w:cols w:space="720"/>'
    s += '<w:docGrid w:linePitch="360"/></w:sectPr>'
    return '<w:p><w:pPr><w:spacing w:line="20" w:lineRule="exact"/>{}<w:rPr>{}</w:rPr></w:pPr></w:p>'.format(s, FONT)


def cell(text, w, header=False, align="center", bold=False, last=False):
    border = '<w:tcBorders><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tcBorders>' if header else ""
    jc = '<w:jc w:val="center"/>' if align == "center" else ""
    p = para(AUTO + '<w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>' + jc,
             runs(text, size=16, bold=header or bold))
    return '<w:tc><w:tcPr><w:tcW w:w="{}" w:type="dxa"/>{}<w:vAlign w:val="center"/></w:tcPr>{}</w:tc>'.format(w, border, p)


def TABLE(widths, rows, bold_rows=(), first_left=True):
    total = sum(widths)
    t = ('<w:tbl><w:tblPr><w:tblW w:w="{}" w:type="dxa"/><w:jc w:val="center"/>'
         '<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
         '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/></w:tblBorders>'
         '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="57" w:type="dxa"/><w:right w:w="57" w:type="dxa"/></w:tblCellMar>'
         '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
         '</w:tblPr><w:tblGrid>').format(total)
    t += "".join('<w:gridCol w:w="{}"/>'.format(w) for w in widths) + "</w:tblGrid>"
    for i, row in enumerate(rows):
        hdr = i == 0
        trpr = '<w:trPr><w:cantSplit/>{}<w:jc w:val="center"/></w:trPr>'.format("<w:tblHeader/>" if hdr else "")
        cells = "".join(cell(c, w, header=hdr, bold=(i in bold_rows),
                             align=("left" if (j == 0 and first_left and not hdr) else "center"))
                        for j, (c, w) in enumerate(zip(row, widths)))
        t += "<w:tr>" + trpr + cells + "</w:tr>"
    return t + "</w:tbl>"


IMAGES = []  # (rid, filename)


def IMG(path, width_twips):
    im = Image.open(path)
    n = len(IMAGES) + 1
    rid, name = "rIdFig{}".format(n), "fig{}.png".format(n)
    IMAGES.append((rid, name, path))
    cx = int(width_twips * EMU_PER_TWIP)
    cy = int(cx * im.height / im.width)
    drawing = (
        '<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        '<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        '<wp:docPr id="{id}" name="Figure {n}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        '<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>'
    ).format(cx=cx, cy=cy, id=100 + n, n=n, name=name, rid=rid)
    return para('<w:keepNext/><w:spacing w:before="120" w:after="0"/><w:jc w:val="center"/>',
                '<w:r><w:rPr>{}<w:noProof/></w:rPr>{}</w:r>'.format(FONT, drawing))


# ---------------------------------------------------------------- content

MANUSCRIPT = os.path.join(REPO, "reports/intermediate_draft.md")


def parse_manuscript(path):
    """Turn the hand-written markdown manuscript into docx XML parts. Formatting only, text untouched.

    Supported: '# ' / '## ' headings, paragraphs, <!-- table widths=.. bold=.. caption=".." --> + md table,
    <!-- figure name width=full|N --> + caption line, <!-- full --> ... <!-- /full --> (one-column block),
    <!-- abstract --> / <!-- title2: .. --> / <!-- refs --> blocks.
    """
    src = open(path, encoding="utf8").read()
    title2 = re.search(r"<!-- title2: (.+?) -->", src).group(1).strip()
    abstract = re.search(r"<!-- abstract -->\s*(.+?)\s*<!-- /abstract -->", src, re.S).group(1).strip()
    src = src[src.index("<!-- /abstract -->") + len("<!-- /abstract -->"):]
    lines = src.split("\n")
    parts, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("## "):
            parts.append(H2(ln[3:]))
        elif ln.startswith("# "):
            parts.append(H1(ln[2:]))
        elif ln.strip() == "<!-- full -->":
            parts.append(SECT(2))
        elif ln.strip() == "<!-- /full -->":
            parts.append(SECT(1))
        elif ln.startswith("<!-- table"):
            widths = [int(w) for w in re.search(r"widths=([\d,]+)", ln).group(1).split(",")]
            m = re.search(r"bold=([\d,]+)", ln)
            bold = tuple(int(b) for b in m.group(1).split(",")) if m else ()
            cap = re.search(r'caption="(.+?)"', ln).group(1)
            rows = []
            i += 1
            while i < n and lines[i].startswith("|"):
                if not re.match(r"^\|[-| ]+\|$", lines[i].strip()):
                    # drop only the single padding space so "|   배경 쪽 |" keeps its indent
                    rows.append([(c[1:] if c.startswith(" ") else c).rstrip()
                                 for c in lines[i].strip()[1:-1].split("|")])
                i += 1
            parts.append(CAP(cap, before=120 if sum(widths) > COL_W else 60))
            parts.append(TABLE(widths, rows, bold_rows=bold))
            parts.append(SPACER())
            continue
        elif ln.startswith("<!-- figure"):
            m = re.match(r"<!-- figure (\S+) width=(\S+) -->", ln)
            w = FULL_W if m.group(2) == "full" else int(m.group(2))
            parts.append(IMG(os.path.join(FIG, m.group(1)), w))
            i += 1
            parts.append(FIGCAP(lines[i].strip()))
        elif ln.strip() == "<!-- refs -->":
            i += 1
            while lines[i].strip() != "<!-- /refs -->":
                if lines[i].strip():
                    parts.append(REF(lines[i].strip()))
                i += 1
        else:
            parts.append(P(ln.strip()))
        i += 1
    return title2, abstract, parts

# ---------------------------------------------------------------- assemble

def main():
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    with zipfile.ZipFile(SRC) as z:
        z.extractall(WORK)
    docp = os.path.join(WORK, "word/document.xml")
    x = open(docp, encoding="utf8").read()

    TITLE_LINE2, ABSTRACT, parts = parse_manuscript(MANUSCRIPT)
    head, rest = x.split("<w:body>", 1)
    elems = re.findall(r"(<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>)", rest, re.S)
    cover = elems[:12]  # cover page .. abstract section break (index 11 holds the footer sectPr)

    cover[4] = cover[4].replace("PROPOSAL REPORT", "INTERMEDIATE REPORT")
    assert "재현 및 개선 계획" in cover[6]
    cover[6] = cover[6].replace("재현 및 개선 계획", TITLE_LINE2)
    cover[6] = cover[6].replace("<w:lastRenderedPageBreak/>", "")
    old_abs = re.search(r"<w:t>(본 프로젝트는.*?)</w:t>", cover[10]).group(1)
    cover[10] = cover[10].replace(old_abs, escape(ABSTRACT))

    final_sect = ('<w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="11906" w:h="16838"/>'
                  '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="720" w:footer="567" w:gutter="0"/>'
                  '<w:cols w:num="2" w:space="425"/><w:docGrid w:linePitch="360"/></w:sectPr>')
    new = head + "<w:body>" + "".join(cover) + "".join(parts) + final_sect + "</w:body></w:document>"
    open(docp, "w", encoding="utf8").write(new)

    # images + relationships
    os.makedirs(os.path.join(WORK, "word/media"), exist_ok=True)
    relp = os.path.join(WORK, "word/_rels/document.xml.rels")
    rels = open(relp, encoding="utf8").read()
    add = ""
    for rid, name, path in IMAGES:
        shutil.copy(path, os.path.join(WORK, "word/media", name))
        add += ('<Relationship Id="{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                'Target="media/{}"/>').format(rid, name)
    open(relp, "w", encoding="utf8").write(rels.replace("</Relationships>", add + "</Relationships>"))

    # metadata
    for f in ["docProps/core.xml", "docProps/app.xml"]:
        p = os.path.join(WORK, f)
        s = open(p, encoding="utf8").read().replace("Proposal Report", "Intermediate Report").replace("HW2 proposal report", "HW2 intermediate report")
        open(p, "w", encoding="utf8").write(s)

    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml first
        z.write(os.path.join(WORK, "[Content_Types].xml"), "[Content_Types].xml")
        for root, _, files in os.walk(WORK):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, WORK)
                if arc != "[Content_Types].xml":
                    z.write(full, arc)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
