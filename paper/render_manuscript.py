"""Typeset the editable manuscript as an academic PDF with embedded fonts."""

import html
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, KeepTogether, Flowable, PageBreak)

ROOT=Path(__file__).resolve().parent
OUT=ROOT
import matplotlib
FONTS=Path(matplotlib.get_data_path())/"fonts/ttf"


def inline(text):
    text=html.escape(text)
    text=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",text)
    text=re.sub(r"`(.+?)`",r'<font name="Code">\1</font>',text)
    def link(m):
        url=m.group(0);tail=""
        while url.endswith((".",",",";")):
            tail=url[-1]+tail;url=url[:-1]
        return '<link href="'+url+'" color="#175b55">'+url+'</link>'+tail
    return re.sub(r"https?://[^\s<>]+",link,text)


class Equation(Flowable):
    def __init__(self,path,number,dpi=400):
        super().__init__();self.path=str(path);self.number=number
        w,h=ImageReader(self.path).getSize();self.basew=w*72/dpi;self.baseh=h*72/dpi
    def wrap(self,available_width,available_height):
        self.width=available_width;scale=min(1.,(available_width-38)/self.basew)
        self.iw=self.basew*scale;self.ih=self.baseh*scale;self.height=self.ih+13
        return self.width,self.height
    def draw(self):
        self.canv.drawImage(self.path,(self.width-32-self.iw)/2,6,width=self.iw,height=self.ih,mask="auto")
        self.canv.setFont("Serif",9)
        self.canv.drawRightString(self.width,self.height/2-2,f"({self.number})")


class Article(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if hasattr(flowable,"outline_level"):
            key="section_"+str(self.seq.nextf("section"))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(),key,level=flowable.outline_level,closed=False)


def main():
    for name,file in (("Serif","DejaVuSerif.ttf"),("SerifBold","DejaVuSerif-Bold.ttf"),
                      ("SerifItalic","DejaVuSerif-Italic.ttf"),("SerifBoldItalic","DejaVuSerif-BoldItalic.ttf"),
                      ("Sans","DejaVuSans.ttf"),("SansBold","DejaVuSans-Bold.ttf"),("Code","DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name,str(FONTS/file)))
    pdfmetrics.registerFontFamily("Serif",normal="Serif",bold="SerifBold",italic="SerifItalic",boldItalic="SerifBoldItalic")
    pdfmetrics.registerFontFamily("Sans",normal="Sans",bold="SansBold",italic="Sans",boldItalic="SansBold")
    styles={
      "body":ParagraphStyle("body",fontName="Serif",fontSize=10.5,leading=14.4,spaceAfter=7.2,
                            alignment=TA_LEFT,allowWidows=0,allowOrphans=0,splitLongWords=True),
      "title":ParagraphStyle("title",fontName="SerifBold",fontSize=21,leading=24.5,spaceAfter=7,keepWithNext=True),
      "subtitle":ParagraphStyle("subtitle",fontName="Serif",fontSize=14.5,leading=18,spaceAfter=15,keepWithNext=True),
      "author":ParagraphStyle("author",fontName="Serif",fontSize=12,leading=16,spaceAfter=7,keepWithNext=True),
      "meta":ParagraphStyle("meta",fontName="Sans",fontSize=8.5,leading=12,textColor=colors.HexColor("#555555"),spaceAfter=16),
      "h1":ParagraphStyle("h1",fontName="SerifBold",fontSize=13,leading=17,spaceBefore=11,spaceAfter=7,keepWithNext=True),
      "h2":ParagraphStyle("h2",fontName="SerifBold",fontSize=11.2,leading=14.8,spaceBefore=9,spaceAfter=6,keepWithNext=True),
      "caption":ParagraphStyle("caption",fontName="Sans",fontSize=8.7,leading=11.8,spaceAfter=9,spaceBefore=5),
      "table":ParagraphStyle("table",fontName="Serif",fontSize=9.1,leading=12),
      "tablehead":ParagraphStyle("tablehead",fontName="SansBold",fontSize=8.6,leading=11.5),
      "ref":ParagraphStyle("ref",fontName="Serif",fontSize=9,leading=12.4,spaceAfter=7,leftIndent=16,firstLineIndent=-16,splitLongWords=True),
    }
    width,height=A4;body_width=width-108
    pdf=OUT/"manuscript.pdf"
    doc=Article(str(pdf),pagesize=A4,leftMargin=54,rightMargin=54,topMargin=50,bottomMargin=51,
                title="Cumulative Stabilization Probabilities During PPO Learning: A Paired Reward Study for Cart-Pendulum Swing-Up",
                author="Shaikhislam Askanbay",subject="Simulation study of paired reward effects on first sustained stabilization",
                pageCompression=1)
    def page(canvas,document):
        canvas.saveState()
        if document.page>1:
            canvas.setFont("Sans",7.5);canvas.setFillColor(colors.HexColor("#666666"))
            canvas.drawString(54,height-29,"CUMULATIVE STABILIZATION PROBABILITIES DURING PPO LEARNING")
            canvas.drawRightString(width-54,height-29,"Askanbay")
            canvas.setStrokeColor(colors.HexColor("#cccccc"));canvas.setLineWidth(.35)
            canvas.line(54,height-35,width-54,height-35)
        canvas.setFont("Sans",7.5);canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(54,29,"Research preprint  |  Version 1.1  |  30 October 2026")
        canvas.drawRightString(width-54,29,str(document.page))
        canvas.restoreState()
    lines=(OUT/"manuscript.md").read_text(encoding="utf-8").splitlines()
    eqs=json.loads((OUT/"equation_index.json").read_text())
    story=[];plates=[];i=0;equation_number=0;references=False;main_started=False
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=="$$":
            i+=1
            while lines[i].strip()!="$$":i+=1
            equation_number+=1
            story.append(Equation(OUT/eqs[equation_number-1]["file"],equation_number));i+=1;continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                row=lines[i].strip()
                if not re.match(r"^\|[\s|:\-]+\|$",row):
                    rows.append([cell.strip() for cell in row.strip("|").split("|")])
                i+=1
            n=len(rows[0]);weights=[]
            for j in range(n):
                length=max(len(row[j]) for row in rows)
                weights.append(max(11,min(38,length))**.75)
            widths=[body_width*w/sum(weights) for w in weights]
            data=[[Paragraph(inline(cell),styles["tablehead" if r==0 else "table"]) for cell in row] for r,row in enumerate(rows)]
            table=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),
                ("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4),
                ("BOTTOMPADDING",(0,0),(-1,-1),4),("LINEABOVE",(0,0),(-1,0),.8,colors.black),
                ("LINEBELOW",(0,0),(-1,0),.6,colors.black),("LINEBELOW",(0,-1),(-1,-1),.6,colors.black),
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f0f1f0"))]))
            if rows[0][0]=="Setting":
                story.extend([Spacer(1,4),table,Spacer(1,9)])
            else:
                story.append(KeepTogether([Spacer(1,4),table,Spacer(1,9)]))
            continue
        match=re.match(r"!\[(.*?)\]\((.*?)\)$",line)
        if match:
            caption,path=match.groups();image_path=OUT/path
            iw,ih=ImageReader(str(image_path)).getSize();dw=body_width;dh=dw*ih/iw
            maximum=height-50-51-75
            if dh>maximum:dw*=maximum/dh;dh=maximum
            image=Image(str(image_path),width=dw,height=dh,hAlign="CENTER")
            block=KeepTogether([Spacer(1,4),image,Paragraph(inline(caption),styles["caption"])])
            if path.endswith("physical_system.png"):story.append(block)
            else:plates.append(block)
            i+=1;continue
        if line.startswith("# "):
            p=Paragraph(inline(line[2:]),styles["title"]);p.outline_level=0;story.append(p);i+=1;continue
        if line.startswith("### "):
            key="h2" if main_started else "subtitle"
            p=Paragraph(inline(line[4:]),styles[key])
            if main_started:p.outline_level=2
            story.append(p);i+=1;continue
        if line.startswith("## "):
            main_started=True;heading=line[3:];references=heading=="References"
            if heading.startswith("6. Discussion") and plates:
                story.extend([PageBreak(),plates[0],PageBreak(),*plates[1:],PageBreak()])
                plates=[]
            p=Paragraph(inline(heading),styles["h1"]);p.outline_level=1;story.append(p);i+=1;continue
        if line=="**Shaikhislam Askanbay**":
            story.append(Paragraph("Shaikhislam Askanbay",styles["author"]));i+=1;continue
        if line.startswith("Research preprint"):
            story.append(Paragraph(inline(line),styles["meta"]));i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(("#","|","![","$$")):
            paragraph.append(lines[i].strip());i+=1
        style=styles["ref" if references else "body"]
        story.append(Paragraph(inline(" ".join(paragraph)),style))
    assert equation_number==len(eqs)
    doc.build(story,onFirstPage=page,onLaterPages=page)
    from pypdf import PdfReader
    reader=PdfReader(pdf);text="\n".join(p.extract_text() for p in reader.pages)
    assert "Shaikhislam Askanbay" in text and "0.0226288" in text
    assert "\ufffd" not in text
    (OUT/"pdf_text_extract.txt").write_text(text,encoding="utf-8")
    (OUT/"pdf_structure_check.json").write_text(json.dumps({"pages":len(reader.pages),"author":reader.metadata.author,
        "title":reader.metadata.title,"equations":equation_number,"embedded_text_characters":len(text),
        "replacement_character_absent":True},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"pdf":str(pdf),"pages":len(reader.pages),"equations":equation_number}))


if __name__=="__main__":main()
