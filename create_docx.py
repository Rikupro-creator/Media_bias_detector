import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_document():
    doc = docx.Document()
    
    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Styling Colors
    NAVY = RGBColor(31, 78, 120)     # #1F4E78 - Dominant Header Color
    SLATE = RGBColor(89, 89, 89)     # #595959 - Subtitles & Secondary Text
    DARK_TEXT = RGBColor(38, 38, 38) # #262626 - Body Text
    WHITE = RGBColor(255, 255, 255)
    
    # Helper functions
    def set_cell_background(cell, fill_hex):
        shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        cell._tc.get_or_add_tcPr().append(shading_elm)

    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
            node = OxmlElement(f'w:{m}')
            node.set(qn('w:w'), str(val))
            node.set(qn('w:type'), 'dxa')
            tcMar.append(node)
        tcPr.append(tcMar)

    def set_table_borders(table, color="D3D3D3"):
        tblPr = table._tbl.tblPr
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>\n'
            f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>\n'
            f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/>\n'
            f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>\n'
            f'  <w:insideV w:val="none"/>\n'
            f'  <w:left w:val="none"/>\n'
            f'  <w:right w:val="none"/>\n'
            f'</w:tblBorders>'
        )
        tblPr.append(borders)

    # Base Normal Style
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = DARK_TEXT

    # Title Block
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("Kenyan Political News Dataset Pipeline & Empirical Analysis")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(16)
    run_sub = sub_p.add_run("Technical Documentation, Script Mechanics, and Stage-by-Stage Descriptive Analysis\nTracing Data Evolution from Raw Scraping (kenyamoja_news.csv) to Verified Bias Annotations (political_verified.parquet)")
    run_sub.font.name = 'Calibri'
    run_sub.font.size = Pt(12.5)
    run_sub.font.italic = True
    run_sub.font.color.rgb = SLATE

    # Divider Line
    divider = doc.add_paragraph()
    divider.paragraph_format.space_after = Pt(16)
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="12" w:space="1" w:color="1F4E78"/></w:pBdr>')
    divider._p.get_or_add_pPr().append(pBdr)

    # Section Heading Helpers
    def add_heading_1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(8)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = NAVY
        return h

    def add_heading_2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(12.5)
        r.font.bold = True
        r.font.color.rgb = NAVY
        return h

    def add_heading_3(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = SLATE
        return h

    def add_callout(text, title="KEY TAKEAWAY"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        set_cell_background(cell, "F2F4F8")
        set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
        
        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>\n'
            f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="1F4E78"/>\n'
            f'  <w:top w:val="none"/>\n'
            f'  <w:right w:val="none"/>\n'
            f'  <w:bottom w:val="none"/>\n'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        r_title = p.add_run(f"📌 {title}\n")
        r_title.font.bold = True
        r_title.font.size = Pt(10)
        r_title.font.color.rgb = NAVY

        r_text = p.add_run(text)
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = DARK_TEXT

        sp = doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY
    # -------------------------------------------------------------
    add_heading_1("1. Executive Summary & Pipeline Architecture")
    
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.add_run(
        "This research report provides a complete technical documentation and empirical descriptive analysis "
        "of the data pipeline developed to construct the Kenyan Political News Media Framing & Bias Dataset. "
        "The codebase comprises 18 Python scripts spanning web scraping, multi-threaded text extraction, "
        "schema integration, zero-shot political subject classification, structured LLM media bias annotation, "
        "and dual-model inter-annotator verification.\n\n"
        "Across 5 distinct processing stages, the dataset matures from 73,020 raw article links harvested from KenyaMoja "
        "into a grand master dataset of 63,939 processed articles, narrows to 16,367 primary political news articles, "
        "and completes with 16,367 dual-model annotated and verified articles in "
    )
    p.add_run("political_verified.parquet").font.bold = True
    p.add_run(" (and export format ").font.italic = True
    p.add_run("political_verified.csv").font.bold = True
    p.add_run(").")

    add_callout(
        "Pipeline Scale & Conversion Funnel:\n"
        "• Stage 1 (Raw Links): 73,020 records across 9 news portals in kenyamoja_news.csv\n"
        "• Stage 2 (Master Dataset): 63,939 unique processed articles in data.parquet\n"
        "• Stage 2 (Political Selection): 16,367 primary political news articles (25.6% of master pool) in political.parquet\n"
        "• Stage 3 (Bias Annotation): 16,367 annotated articles via OpenAI (GPT-5.6-Luna) in political_annotated.parquet\n"
        "• Stage 4 (Judge Verification): 16,367 verified articles via Google Gemini 3.1 in political_verified.parquet",
        "PIPELINE CONVERSION METRICS"
    )

    # -------------------------------------------------------------
    # SECTION 2: PYTHON SCRIPTS INVENTORY TABLE
    # -------------------------------------------------------------
    add_heading_1("2. Python Scripts Inventory & Operational Summary")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.add_run(
        "The table below inventories all 18 Python scripts in the project directory, detailing their functional role, "
        "inputs, outputs, primary logic, and execution status."
    )

    table_data = [
        ("Script Name", "Category / Phase", "Input File(s)", "Output File(s)", "Primary Function & Logic", "Status"),
        (
            "scraper.py",
            "1. Raw Scraping",
            "KenyaMoja URLs,\nprogress.json",
            "kenyamoja_news.csv,\nprogress.json",
            "Resumable Selenium Chrome scraper crawling 9 KenyaMoja sources up to page 200. Saves metadata.",
            "Active"
        ),
        (
            "paragraphs.py",
            "1. Text Extraction",
            "kenyamoja_news.csv",
            "kenyamoja_news_with_\narticles.parquet,\nfailed_links.csv",
            "Multi-threaded requests/bs4 scraper extracting article paragraph text (<p>) and logging failed URLs.",
            "Active"
        ),
        (
            "failed_urls.py",
            "1. Link Recovery",
            "failed_links.csv,\nkenyamoja_news_with_\narticles.csv",
            "kenyamoja_news_with_\narticles.csv,\nstill_failed_after_\nselenium.csv",
            "Headless Edge Selenium retry worker extracting body text from failed URLs and updating main CSV.",
            "Active"
        ),
        (
            "scraper2.py",
            "1. Text Extraction",
            "kenyamoja_news.csv",
            "kenyamoja_articles.csv,\nkenyamoja_skipped.csv",
            "Multi-site article parser with site-specific rules (requests for static sites; Chrome for dynamic sites).",
            "Active"
        ),
        (
            "scraper3.py",
            "1. Text Extraction",
            "kenyamoja_news.csv",
            "kenyamoja_selenium_\narticles.csv,\nkenyamoja_selenium_\nskipped.csv",
            "Dedicated thread-pool Edge Selenium scraper targeting JS-rendered sites (Nation Africa & K24).",
            "Active"
        ),
        (
            "paragraphs2.py",
            "1. Testing / Helper",
            "Hardcoded URL",
            "Console Stdout",
            "Single-page prototype script testing HTML parsing selectors for Citizen Digital articles.",
            "Helper / Test"
        ),
        (
            "conver.py",
            "2. Format Conversion",
            "kenyamoja_articles.csv",
            "kenyamoja_articles1.\nparquet",
            "Utility script converting raw scraped CSV articles into Parquet format for fast processing.",
            "Active"
        ),
        (
            "cleaner.py",
            "2. Diagnostics",
            "kenyamoja_subset.parquet,\nkenyamoja_articles.csv",
            "Console Stdout",
            "Diagnostic utility comparing column structures and previewing head rows across scraped datasets.",
            "Diagnostic"
        ),
        (
            "analysis.py",
            "2. Diagnostics",
            "kenyamoja_articles1.\nparquet",
            "Console Stdout",
            "Diagnostic utility printing missing value counts across columns in intermediate Parquet files.",
            "Diagnostic"
        ),
        (
            "classifier.py",
            "2. Classification",
            "kenyamoja_subset.\nparquet",
            "kenyan_news_political_\nclassified1.csv / .parquet",
            "Classifies subset parquet articles as Political vs Not Political using OpenAI (gpt-5.6-luna).",
            "Active"
        ),
        (
            "comb.py",
            "2. Classification",
            "kenyamoja_articles1.\nparquet, final1.parquet",
            "final.parquet,\nkenyan_news_political_\nclassified.parquet / .csv",
            "Concatenates scraped subsets into final.parquet and classifies political relevance with gpt-5.6-luna.",
            "Active"
        ),
        (
            "comb2.py",
            "2. Integration",
            "classified.parquet,\nclassified1.parquet",
            "data.parquet,\npolitical.parquet",
            "Standardizes column schemas, merges subsets into data.parquet, and extracts political news into political.parquet.",
            "Active"
        ),
        (
            "annotator.py",
            "3. Bias Annotation",
            "political.parquet",
            "political_annotated.\nparquet",
            "Annotates political articles for media framing bias (biased / not_biased) + evidence using gpt-5.6-luna.",
            "Active"
        ),
        (
            "judge.py",
            "4. LLM Verification",
            "political_annotated.\nparquet",
            "political_verified.parquet,\njudge_disagreements.csv,\njudge_metrics.json",
            "Independent LLM judge (gemini-3.1-flash-lite) verifying bias labels, verbatim quotes, and agreement metrics.",
            "Active"
        ),
        (
            "change.py",
            "5. Export",
            "political_verified.\nparquet",
            "political_verified.csv",
            "Converts the final verified Parquet research dataset to CSV format for external inspection.",
            "Active"
        ),
        (
            "data_analyzer.py",
            "5. Data Analysis",
            "political_verified.\nparquet",
            "Console Stdout",
            "Computes statistical summaries (publisher breakdown, political labels, judge agreement counts).",
            "Active"
        ),
        (
            "verifier.py",
            "N/A",
            "None",
            "None",
            "Empty placeholder script in repository.",
            "Placeholder"
        ),
        (
            "final_dataset.py",
            "N/A",
            "None",
            "None",
            "Empty placeholder script in repository.",
            "Placeholder"
        ),
    ]

    table = doc.add_table(rows=len(table_data), cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    col_widths = [Inches(1.1), Inches(1.0), Inches(1.2), Inches(1.3), Inches(1.7), Inches(0.7)]
    set_table_borders(table)

    for row_idx, row_content in enumerate(table_data):
        row = table.rows[row_idx]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if row_idx == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        is_header = (row_idx == 0)

        for col_idx, cell_value in enumerate(row_content):
            cell = row.cells[col_idx]
            cell.width = col_widths[col_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)

            if is_header:
                set_cell_background(cell, "1F4E78")
            else:
                set_cell_background(cell, "FFFFFF" if row_idx % 2 == 1 else "F9FAFC")

            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(cell_value)
            run.font.name = 'Calibri'
            if is_header:
                run.font.bold = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = WHITE
            else:
                run.font.size = Pt(8.5)
                run.font.color.rgb = DARK_TEXT

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # SECTION 3: STAGE-BY-STAGE EMPIRICAL DESCRIPTIVE ANALYSIS
    # -------------------------------------------------------------
    add_heading_1("3. Step-by-Step Data Lineage & Stage Descriptive Analysis")

    p = doc.add_paragraph()
    p.add_run(
        "This section walks step-by-step through each stage of the pipeline, providing both the procedural data "
        "flow narrative and empirical descriptive statistics for the dataset generated at each stage."
    )

    # STAGE 1
    add_heading_2("Stage 1: Raw Web Scraping & Article Text Harvesting")
    
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "1. "
    )
    p.add_run("Raw Link Scraping (scraper.py): ").font.bold = True
    p.add_run(
        "The pipeline initiates with scraper.py, an automated Chrome Selenium crawler targeting the KenyaMoja news "
        "aggregator portal across 9 distinct Kenyan media channels. It harvests metadata (article title, published date, "
        "publisher source, canonical URL) across up to 200 pages per portal and outputs "
    )
    p.add_run("kenyamoja_news.csv").font.bold = True
    p.add_run(".\n\n"
              "2. ")
    p.add_run("Static & Dynamic Body Text Extraction (paragraphs.py, scraper2.py, scraper3.py, failed_urls.py): ").font.bold = True
    p.add_run(
        "To fetch full article body text, paragraphs.py and scraper2.py parse HTML paragraph elements (<p>) using concurrent HTTP "
        "requests. Because client-side JavaScript portals like Nation Africa (nation.africa) and K24 (k24.digital) fail under "
        "static GET requests, failed_urls.py and scraper3.py deploy thread-pool Edge Selenium webdrivers to render dynamic HTML "
        "and recover full article texts.\n\n"
        "3. "
    )
    p.add_run("Parquet Standardization (conver.py): ").font.bold = True
    p.add_run(
        "conver.py converts scraped CSV outputs into "
    )
    p.add_run("kenyamoja_articles1.parquet").font.bold = True
    p.add_run(" for optimized memory storage and high-speed retrieval.")

    add_heading_3("Empirical Descriptive Analysis — Stage 1 Datasets")
    
    # Stage 1 Table 1: kenyamoja_news.csv source breakdown
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.add_run("• ").font.bold = True
    p.add_run("kenyamoja_news.csv Descriptive Breakdown (N = 73,020 records, 4 columns: name, date, article_title, url; 0 missing values):")

    s1_data = [
        ("Publisher Source (name)", "Raw Article Count", "Percentage of Total (%)", "Harvested Data Quality"),
        ("Citizen Digital (citizen)", "10,050", "13.76%", "100% complete metadata, 0 nulls"),
        ("The Star Kenya (star)", "10,050", "13.76%", "100% complete metadata, 0 nulls"),
        ("Tuko Kenya (tuko)", "10,050", "13.76%", "100% complete metadata, 0 nulls"),
        ("The Standard (standard)", "10,050", "13.76%", "100% complete metadata, 0 nulls"),
        ("K24 Digital (k24)", "10,022", "13.72%", "100% complete metadata, 0 nulls"),
        ("Capital FM Kenya (capital-fm)", "9,083", "12.44%", "100% complete metadata, 0 nulls"),
        ("Nation Africa (nation)", "8,983", "12.30%", "100% complete metadata, 0 nulls"),
        ("TV47 Kenya (tv47)", "2,817", "3.86%", "100% complete metadata, 0 nulls"),
        ("KBC Channel 1 (kbc)", "1,915", "2.62%", "100% complete metadata, 0 nulls"),
        ("Total Harvested", "73,020", "100.00%", "Complete metadata pool for text extraction"),
    ]

    tbl_s1 = doc.add_table(rows=len(s1_data), cols=4)
    tbl_s1.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_s1.autofit = False
    set_table_borders(tbl_s1)
    w_s1 = [Inches(2.2), Inches(1.3), Inches(1.4), Inches(1.6)]

    for r_i, r_c in enumerate(s1_data):
        row = tbl_s1.rows[r_i]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if r_i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        is_h = (r_i == 0 or r_i == len(s1_data)-1)

        for c_i, val in enumerate(r_c):
            cell = row.cells[c_i]
            cell.width = w_s1[c_i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            if is_h:
                set_cell_background(cell, "1F4E78" if r_i == 0 else "EBF1F5")
            else:
                set_cell_background(cell, "FFFFFF" if r_i % 2 == 1 else "F9FAFC")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = 'Calibri'
            if is_h:
                run.font.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = WHITE if r_i == 0 else NAVY
            else:
                run.font.size = Pt(8.5)
                run.font.color.rgb = DARK_TEXT

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.add_run("• ").font.bold = True
    p.add_run("kenyamoja_articles1.parquet Summary (N = 1,064 records, 4 columns: url, website, num_paragraphs, article_text):\n")
    p.add_run("  - Website scope: 100% Capital FM articles (1,064 rows).\n")
    p.add_run("  - Paragraph count distribution (num_paragraphs): Mean = 13.87, Std = 7.56, Median = 13, IQR = [10, 17], Min = 0, Max = 72.\n\n")
    p.add_run("• ").font.bold = True
    p.add_run("kenyamoja_subset.parquet Summary (N = 17,851 records, 9 columns, 0 missing values):\n")
    p.add_run("  - Publisher composition: K24 Digital = 10,022 articles (56.14%), Capital FM = 7,829 articles (43.86%).")

    # STAGE 2
    add_heading_2("Stage 2: Dataset Consolidation & Zero-Shot Political Classification")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "1. "
    )
    p.add_run("Dataset Union & Master Assembly (comb.py & classifier.py): ").font.bold = True
    p.add_run(
        "The scraped text partitions (kenyamoja_articles1.parquet, final1.parquet, and kenyamoja_subset.parquet) are concatenated vertically. "
        "comb.py generates final.parquet (46,088 rows).\n\n"
        "2. "
    )
    p.add_run("Zero-Shot Political Classification (comb.py & classifier.py): ").font.bold = True
    p.add_run(
        "Both scripts process article body texts through OpenAI API (gpt-5.6-luna) with multithreaded workers (10 workers). "
        "A strict definition prompt instructs the model to classify an article as 'Political' ONLY if its primary subject concerns "
        "Kenyan political actors, political institutions, elections, legislation, governance, or government accountability.\n\n"
        "3. "
    )
    p.add_run("Master Schema Integration & Political Extraction (comb2.py): ").font.bold = True
    p.add_run(
        "comb2.py standardizes column names across classified subsets, merges them into a master pool "
    )
    p.add_run("data.parquet").font.bold = True
    p.add_run(" (63,939 total unique articles), and filters for rows where political_classification == 'Political'. "
              "This isolates exactly ")
    p.add_run("16,367 primary political news articles").font.bold = True
    p.add_run(" into ")
    p.add_run("political.parquet").font.bold = True
    p.add_run(".")

    add_heading_3("Empirical Descriptive Analysis — Stage 2 Datasets")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.add_run("• ").font.bold = True
    p.add_run("Master Dataset data.parquet Classification Distribution (N = 63,939 records, 4 columns: url, website, article_text, political_classification):")

    s2_class_data = [
        ("Political Classification Category", "Article Count", "Percentage of Total (%)", "Category Description & Handling"),
        ("Not Political", "35,892", "56.13%", "Articles primarily covering sports, business, health, entertainment, crime, or human interest."),
        ("Political", "16,367", "25.60%", "Primary political news articles extracted into political.parquet for bias annotation."),
        ("Empty", "10,815", "16.91%", "URLs where body text was empty or unextractable prior to classification."),
        ("Error", "865", "1.35%", "API rate limit timeouts or parsing errors during classification pass."),
        ("Total Master Pool", "63,939", "100.00%", "Grand total processed article pool in data.parquet"),
    ]

    tbl_s2_class = doc.add_table(rows=len(s2_class_data), cols=4)
    tbl_s2_class.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_s2_class.autofit = False
    set_table_borders(tbl_s2_class)
    w_s2_c = [Inches(1.8), Inches(1.1), Inches(1.3), Inches(2.3)]

    for r_i, r_c in enumerate(s2_class_data):
        row = tbl_s2_class.rows[r_i]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if r_i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        is_h = (r_i == 0 or r_i == len(s2_class_data)-1)

        for c_i, val in enumerate(r_c):
            cell = row.cells[c_i]
            cell.width = w_s2_c[c_i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            if is_h:
                set_cell_background(cell, "1F4E78" if r_i == 0 else "EBF1F5")
            else:
                set_cell_background(cell, "FFFFFF" if r_i % 2 == 1 else "F9FAFC")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = 'Calibri'
            if is_h:
                run.font.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = WHITE if r_i == 0 else NAVY
            else:
                run.font.size = Pt(8.5)
                run.font.color.rgb = DARK_TEXT

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.add_run("• ").font.bold = True
    p.add_run("Extracted Political Dataset political.parquet Breakdown by News Publisher (N = 16,367 political articles):")

    s2_pub_data = [
        ("Publisher Website", "Master Pool (data.parquet)", "Political Articles (political.parquet)", "Political Extraction Rate (%)"),
        ("Capital FM", "9,083", "3,578", "39.39%"),
        ("The Standard", "10,043", "3,435", "34.20%"),
        ("The Star Kenya", "10,050", "3,127", "31.11%"),
        ("Citizen Digital", "10,044", "3,090", "30.76%"),
        ("K24 Digital", "10,022", "1,385", "13.82%"),
        ("TV47 Kenya", "2,812", "1,268", "45.09%"),
        ("KBC Channel 1", "1,888", "427", "22.62%"),
        ("Tuko Kenya", "9,997", "57", "0.57%"),
        ("Total Dataset", "63,939", "16,367", "25.60%"),
    ]

    tbl_s2_pub = doc.add_table(rows=len(s2_pub_data), cols=4)
    tbl_s2_pub.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_s2_pub.autofit = False
    set_table_borders(tbl_s2_pub)
    w_s2_p = [Inches(1.8), Inches(1.5), Inches(1.7), Inches(1.5)]

    for r_i, r_c in enumerate(s2_pub_data):
        row = tbl_s2_pub.rows[r_i]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if r_i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        is_h = (r_i == 0 or r_i == len(s2_pub_data)-1)

        for c_i, val in enumerate(r_c):
            cell = row.cells[c_i]
            cell.width = w_s2_p[c_i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            if is_h:
                set_cell_background(cell, "1F4E78" if r_i == 0 else "EBF1F5")
            else:
                set_cell_background(cell, "FFFFFF" if r_i % 2 == 1 else "F9FAFC")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = 'Calibri'
            if is_h:
                run.font.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = WHITE if r_i == 0 else NAVY
            else:
                run.font.size = Pt(8.5)
                run.font.color.rgb = DARK_TEXT

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.add_run("• ").font.bold = True
    p.add_run("political.parquet Article Word Length Summary Statistics (N = 16,367 articles):\n")
    p.add_run("  - Mean length: 890.67 words | Std Dev: 1,610.13 words\n")
    p.add_run("  - Median length: 422 words | Interquartile Range (IQR): [287 words (25th percentile), 576 words (75th percentile)]\n")
    p.add_run("  - Minimum length: 2 words | Maximum length: 19,922 words.")

    # STAGE 3
    add_heading_2("Stage 3: Media Bias Framing Annotation (GPT-5.6-Luna)")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "1. "
    )
    p.add_run("Structured Bias Annotation (annotator.py): ").font.bold = True
    p.add_run(
        "annotator.py processes political.parquet (16,367 rows) through OpenAI API (gpt-5.6-luna) using JSON Schema enforcement. "
        "The model analyzes media framing mechanisms (loaded language, asymmetric treatment, selective emphasis, source omission) "
        "and forces a binary decision:\n"
        "• biased: framing favors or disadvantages a political actor/perspective, with evidence detailed in bias_reason.\n"
        "• not_biased: reporting is factual, neutral, or properly attributes political accusations to sources.\n\n"
        "Output: "
    )
    p.add_run("political_annotated.parquet").font.bold = True
    p.add_run(" (16,367 rows, columns: url, website, article_text, political_classification, bias_classification, bias_reason).")

    add_heading_3("Empirical Descriptive Analysis — Stage 3 Bias Annotations")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.add_run("• ").font.bold = True
    p.add_run("GPT-5.6-Luna Media Bias Classification Distribution across News Publishers (N = 16,367 political articles):")

    s3_bias_data = [
        ("Publisher Website", "Total Political Articles", "Not Biased Count (%)", "Biased Count (%)", "Publisher Bias Rate (%)"),
        ("Capital FM", "3,578", "2,563 (71.63%)", "1,015 (28.37%)", "28.37%"),
        ("Citizen Digital", "3,090", "2,020 (65.37%)", "1,070 (34.63%)", "34.63%"),
        ("K24 Digital", "1,385", "721 (52.06%)", "664 (47.94%)", "47.94%"),
        ("KBC Channel 1", "427", "232 (54.33%)", "195 (45.67%)", "45.67%"),
        ("The Standard", "3,435", "1,832 (53.33%)", "1,603 (46.67%)", "46.67%"),
        ("The Star Kenya", "3,127", "2,386 (76.30%)", "741 (23.70%)", "23.70%"),
        ("TV47 Kenya", "1,268", "702 (55.36%)", "566 (44.64%)", "44.64%"),
        ("Tuko Kenya", "57", "25 (43.86%)", "32 (56.14%)", "56.14%"),
        ("Overall Dataset", "16,367", "10,481 (64.04%)", "5,886 (35.96%)", "35.96%"),
    ]

    tbl_s3_bias = doc.add_table(rows=len(s3_bias_data), cols=5)
    tbl_s3_bias.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_s3_bias.autofit = False
    set_table_borders(tbl_s3_bias)
    w_s3_b = [Inches(1.6), Inches(1.1), Inches(1.3), Inches(1.2), Inches(1.3)]

    for r_i, r_c in enumerate(s3_bias_data):
        row = tbl_s3_bias.rows[r_i]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if r_i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        is_h = (r_i == 0 or r_i == len(s3_bias_data)-1)

        for c_i, val in enumerate(r_c):
            cell = row.cells[c_i]
            cell.width = w_s3_b[c_i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            if is_h:
                set_cell_background(cell, "1F4E78" if r_i == 0 else "EBF1F5")
            else:
                set_cell_background(cell, "FFFFFF" if r_i % 2 == 1 else "F9FAFC")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = 'Calibri'
            if is_h:
                run.font.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = WHITE if r_i == 0 else NAVY
            else:
                run.font.size = Pt(8.5)
                run.font.color.rgb = DARK_TEXT

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(6)

    # STAGE 4
    add_heading_2("Stage 4: LLM Judge Verification & Inter-Model Audit (Gemini 3.1 Flash Lite)")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "1. "
    )
    p.add_run("Independent Multi-Stage Judge Audit (judge.py): ").font.bold = True
    p.add_run(
        "judge.py conducts a second-pass independent verification using Google Gemini API (gemini-3.1-flash-lite, thinking level high). "
        "The judge performs a 2-stage analysis:\n"
        "  • Stage 1: Independent framing assessment without looking at Luna's label.\n"
        "  • Stage 2: Verification of Luna's label and justification against verbatim textual quotes.\n\n"
        "2. "
    )
    p.add_run("Agreement Metrics & Disagreement Logging: ").font.bold = True
    p.add_run(
        "judge.py calculates inter-model agreement metrics, validates quote grounding against raw text, logs 4,723 disagreement "
        "cases to judge_disagreements.csv, writes summary metrics to judge_agreement_metrics.json, and outputs the final 18-column dataset "
        "political_verified.parquet."
    )

    add_heading_3("Empirical Descriptive Analysis — Stage 4 Inter-Model Verification")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.add_run("• ").font.bold = True
    p.add_run("Cross-Tabulation Matrix: Primary LLM (GPT-5.6-Luna) vs Independent LLM Judge (Gemini 3.1 Flash Lite):")

    s4_matrix_data = [
        ("Primary Annotator (Luna) Label", "Gemini Judge: biased", "Gemini Judge: not_biased", "Total Luna Annotations", "Agreement Classification"),
        ("biased", "1,165", "4,721", "5,886", "1,165 Confirmed Biased | 4,721 Over-flagged"),
        ("not_biased", "2", "10,479", "10,481", "10,479 Confirmed Neutral | 2 Under-flagged"),
        ("Total Gemini Verdicts", "1,167", "15,200", "16,367", "11,644 Total Agreed (71.14%)"),
    ]

    tbl_s4_mat = doc.add_table(rows=len(s4_matrix_data), cols=5)
    tbl_s4_mat.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_s4_mat.autofit = False
    set_table_borders(tbl_s4_mat)
    w_s4_m = [Inches(1.6), Inches(1.1), Inches(1.2), Inches(1.2), Inches(1.4)]

    for r_i, r_c in enumerate(s4_matrix_data):
        row = tbl_s4_mat.rows[r_i]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if r_i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        is_h = (r_i == 0 or r_i == len(s4_matrix_data)-1)

        for c_i, val in enumerate(r_c):
            cell = row.cells[c_i]
            cell.width = w_s4_m[c_i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            if is_h:
                set_cell_background(cell, "1F4E78" if r_i == 0 else "EBF1F5")
            else:
                set_cell_background(cell, "FFFFFF" if r_i % 2 == 1 else "F9FAFC")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = 'Calibri'
            if is_h:
                run.font.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = WHITE if r_i == 0 else NAVY
            else:
                run.font.size = Pt(8.5)
                run.font.color.rgb = DARK_TEXT

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.add_run("• ").font.bold = True
    p.add_run("Gemini Judge Framing Mechanisms Detected (judge_evidence_type breakdown):")

    s4_mech_data = [
        ("Framing Mechanism Category", "Occurrences Count", "Percentage of Evidence Cases (%)", "Operational Definition & Example"),
        ("loaded_language", "853", "34.33%", "Emotive, evaluative, or partisan terms describing political actors or events."),
        ("source_imbalance", "440", "17.71%", "Substantial platform given to one side while ignoring or marginalizing opposition."),
        ("selective_framing", "434", "17.47%", "Structuring story around one political narrative while suppressing counter-narratives."),
        ("negative_characterization", "397", "15.98%", "Systematically portraying a politician or party using hostile descriptions."),
        ("positive_characterization", "178", "7.16%", "Uncritically flattering portrayal of a political figure beyond factual reporting."),
        ("asymmetric_treatment", "151", "6.08%", "Applying unequal standards of scrutiny to government vs opposition claims."),
        ("omission", "103", "4.14%", "Visibly omitting crucial contextual facts or official responses from key parties."),
        ("other", "25", "1.01%", "Unconventional or compound media framing mechanisms."),
        ("Total Cited Mechanisms", "2,485", "100.00%", "Cited across 1,167 judge-classified biased articles"),
    ]

    tbl_s4_mech = doc.add_table(rows=len(s4_mech_data), cols=4)
    tbl_s4_mech.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_s4_mech.autofit = False
    set_table_borders(tbl_s4_mech)
    w_s4_k = [Inches(1.8), Inches(1.1), Inches(1.3), Inches(2.3)]

    for r_i, r_c in enumerate(s4_mech_data):
        row = tbl_s4_mech.rows[r_i]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if r_i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        is_h = (r_i == 0 or r_i == len(s4_mech_data)-1)

        for c_i, val in enumerate(r_c):
            cell = row.cells[c_i]
            cell.width = w_s4_k[c_i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            if is_h:
                set_cell_background(cell, "1F4E78" if r_i == 0 else "EBF1F5")
            else:
                set_cell_background(cell, "FFFFFF" if r_i % 2 == 1 else "F9FAFC")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = 'Calibri'
            if is_h:
                run.font.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = WHITE if r_i == 0 else NAVY
            else:
                run.font.size = Pt(8.5)
                run.font.color.rgb = DARK_TEXT

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(6)

    # STAGE 5
    add_heading_2("Stage 5: Final Formatting & Data Analysis")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "1. "
    )
    p.add_run("CSV Export (change.py): ").font.bold = True
    p.add_run(
        "change.py exports political_verified.parquet to "
    )
    p.add_run("political_verified.csv").font.bold = True
    p.add_run(" (16,367 rows, 18 columns) for external spreadsheet auditing.\n\n"
              "2. ")
    p.add_run("Exploratory Data Analysis (data_analyzer.py): ").font.bold = True
    p.add_run(
        "data_analyzer.py loads political_verified.parquet and prints final summary statistics, confirming publisher distribution, "
        "political article totals, and inter-model agreement counts."
    )

    # -------------------------------------------------------------
    # SECTION 4: DETAILED SCRIPT-BY-SCRIPT MECHANICS
    # -------------------------------------------------------------
    add_heading_1("4. Detailed Script-by-Script Mechanics & Output Explanations")

    p = doc.add_paragraph()
    p.add_run(
        "This section provides an exhaustive technical breakdown of each Python script in the repository, "
        "explaining its execution mechanics, input parameters, generated outputs, and schema details."
    )

    script_details = [
        (
            "scraper.py",
            "Resumable KenyaMoja Scraper",
            "scraper.py connects to KenyaMoja news aggregator base URLs using Selenium Chrome WebDriver. It iterates through 9 news "
            "sources (TV47, Citizen, Star, Tuko, Standard, Nation, Capital FM, K24, KBC) up to page 200 per source. It features "
            "incremental CSV appending, driver auto-restart on crash, and JSON state persistence in progress.json.",
            "KenyaMoja source URLs, progress.json",
            "kenyamoja_news.csv (73,020 rows, columns: name, date, article_title, url), progress.json",
            "Contains raw article metadata scraped from KenyaMoja. Provides the foundational pool of article links for subsequent text extraction."
        ),
        (
            "paragraphs.py",
            "Concurrent Article Paragraph Scraper",
            "paragraphs.py reads article URLs from kenyamoja_news.csv and uses ThreadPoolExecutor (20 workers) with requests and BeautifulSoup "
            "to extract full paragraph text (<p>) from news websites. Employs per-domain semaphores to prevent IP rate-limiting and logs "
            "failed/skipped URLs.",
            "kenyamoja_news.csv",
            "kenyamoja_news_with_articles.parquet, failed_links.csv, still_failed_links.csv",
            "Contains full article body text matched to URLs. Failed/skipped URLs are logged to CSV for recovery."
        ),
        (
            "failed_urls.py",
            "Selenium Edge Failed Link Retry Worker",
            "failed_urls.py processes failed_links.csv using a thread pool of headless Edge webdrivers (4 workers). It renders dynamic HTML, "
            "extracts paragraph text, and merges successfully recovered articles back into the main dataset.",
            "kenyamoja_news_with_articles.csv, failed_links.csv",
            "Updated kenyamoja_news_with_articles.csv, still_failed_after_selenium.csv",
            "Updated main dataset with recovered article text. Unrecoverable URLs are written to still_failed_after_selenium.csv."
        ),
        (
            "scraper2.py",
            "Multi-Site Custom Article Scraper",
            "scraper2.py implements site-specific scraping functions for 9 major Kenyan media portals. Employs HTTP requests for static sites "
            "(Citizen, TV47, Star, Tuko, Standard, Capital FM, KBC) and headless Chrome for dynamic sites (Nation, K24). Filters non-article URLs.",
            "kenyamoja_news.csv",
            "kenyamoja_articles.csv (1,064 rows), kenyamoja_skipped.csv",
            "Contains clean structured article records (columns: url, website, num_paragraphs, article_text)."
        ),
        (
            "scraper3.py",
            "Multi-Threaded Edge Selenium Scraper for Dynamic Portals",
            "scraper3.py strictly targets JavaScript-heavy news sites (Nation Africa & K24 Digital). Uses thread-local persistent Edge webdrivers "
            "with eager page load strategy and CSS wait selectors (div.paragraph-wrapper, p.wp-block-paragraph).",
            "kenyamoja_news.csv",
            "kenyamoja_selenium_articles.csv, kenyamoja_selenium_skipped.csv",
            "Contains recovered articles from dynamic portals that fail standard HTTP GET requests."
        ),
        (
            "paragraphs2.py",
            "Citizen Digital HTML Parser Prototype",
            "paragraphs2.py is a minimal diagnostic script testing BeautifulSoup selector logic (div.article-content__paragraph_XUrcR) on a single "
            "Citizen Digital article URL.",
            "Hardcoded article URL",
            "Console stdout",
            "Verification of CSS selector accuracy for Citizen Digital."
        ),
        (
            "conver.py",
            "CSV to Parquet Format Converter",
            "conver.py reads kenyamoja_articles.csv using pandas and writes it out as kenyamoja_articles1.parquet.",
            "kenyamoja_articles.csv",
            "kenyamoja_articles1.parquet (1,064 rows, columns: url, website, num_paragraphs, article_text)",
            "Optimized Parquet dataset enabling fast I/O and low memory footprint for downstream LLM processing."
        ),
        (
            "cleaner.py",
            "Dataset Schema Inspection Utility",
            "cleaner.py loads kenyamoja_subset.parquet and kenyamoja_articles.csv, printing column names and head preview rows.",
            "kenyamoja_subset.parquet, kenyamoja_articles.csv",
            "Console stdout",
            "Diagnostic output confirming column names and schema compatibility prior to dataset concatenation."
        ),
        (
            "analysis.py",
            "Parquet Missing Value Analyzer",
            "analysis.py loads kenyamoja_articles1.parquet and computes null value counts (df.isna().sum()).",
            "kenyamoja_articles1.parquet",
            "Console stdout",
            "Data quality audit confirming missing value counts across dataset columns."
        ),
        (
            "classifier.py",
            "Zero-Shot Political Classifier for Subset Parquet",
            "classifier.py reads kenyamoja_subset.parquet (17,851 rows) and executes parallel calls to OpenAI API (gpt-5.6-luna) with a structured "
            "political definition prompt. Classifies articles as Political vs Not Political.",
            "kenyamoja_subset.parquet",
            "kenyan_news_political_classified1.csv / .parquet (17,851 rows)",
            "Dataset subset augmented with political_classification labels."
        ),
        (
            "comb.py",
            "Dataset Merger & Zero-Shot Political Classifier",
            "comb.py concatenates scraped article datasets (kenyamoja_articles1.parquet, final1.parquet) into final.parquet (46,088 rows). "
            "It then classifies all articles using OpenAI gpt-5.6-luna with 10 concurrent workers.",
            "kenyamoja_articles1.parquet, final1.parquet",
            "final.parquet (46,088 rows), kenyan_news_political_classified.parquet / .csv (46,088 rows)",
            "Unified master dataset with political_classification labels."
        ),
        (
            "comb2.py",
            "Classified Subsets Standardizer & Political Extractor",
            "comb2.py loads kenyan_news_political_classified.parquet (46,088 rows) and kenyan_news_political_classified1.parquet (17,851 rows), "
            "standardizes column names, concatenates them into data.parquet (63,939 rows), and extracts rows where political_classification == 'Political'.",
            "kenyan_news_political_classified.parquet, kenyan_news_political_classified1.parquet",
            "data.parquet (63,939 rows), political.parquet (16,367 rows)",
            "political.parquet contains all 16,367 primary political articles selected for bias annotation."
        ),
        (
            "annotator.py",
            "LLM Media Bias Framing Annotator",
            "annotator.py processes political.parquet (16,367 rows) through OpenAI API (gpt-5.6-luna) using JSON Schema enforcement. "
            "Annotates articles for framing bias (biased vs not_biased) and generates specific evidence-based justifications (bias_reason).",
            "political.parquet",
            "political_annotated.parquet (16,367 rows, columns: url, website, article_text, political_classification, bias_classification, bias_reason)",
            "Annotated dataset containing primary LLM bias labels and evidence justifications."
        ),
        (
            "judge.py",
            "Google Gemini Dual-Stage Independent LLM Judge & Verifier",
            "judge.py performs an independent second-pass verification on political_annotated.parquet using Google Gemini API (gemini-3.1-flash-lite, "
            "thinking level high). Evaluates independent bias labels, checks verbatim quote grounding in text, logs disagreement cases, and calculates metrics.",
            "political_annotated.parquet",
            "gemini_judge_results.jsonl, judge_disagreements.csv (4,723 rows), judge_agreement_metrics.json, political_verified.parquet (16,367 rows, 18 columns)",
            "Complete final research dataset with independent judge labels, verbatim quote evidence, confidence levels, and agreement status."
        ),
        (
            "change.py",
            "Parquet to CSV Exporter",
            "change.py reads political_verified.parquet and converts it to political_verified.csv using pandas.",
            "political_verified.parquet",
            "political_verified.csv (16,367 rows, 18 columns)",
            "CSV version of the final verified dataset for human spreadsheet inspection."
        ),
        (
            "data_analyzer.py",
            "Final Dataset Exploratory Analyzer",
            "data_analyzer.py loads political_verified.parquet and computes total article counts, publisher volume breakdown, political classification counts, and judge label distributions.",
            "political_verified.parquet",
            "Console stdout",
            "Summary statistics confirming final dataset composition."
        ),
        (
            "verifier.py",
            "Workspace Placeholder Script",
            "Empty script file in the directory.",
            "None",
            "None",
            "Unimplemented placeholder."
        ),
        (
            "final_dataset.py",
            "Workspace Placeholder Script",
            "Empty script file in the directory.",
            "None",
            "None",
            "Unimplemented placeholder."
        ),
    ]

    for script_name, script_title, desc, inputs, outputs, output_exp in script_details:
        add_heading_2(f"{script_name} — {script_title}")
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.add_run("• Technical Mechanics: ").font.bold = True
        p.add_run(desc + "\n")
        p.add_run("• Inputs: ").font.bold = True
        p.add_run(inputs + "\n")
        p.add_run("• Outputs: ").font.bold = True
        p.add_run(outputs + "\n")
        p.add_run("• Output Explanation: ").font.bold = True
        p.add_run(output_exp)

    # -------------------------------------------------------------
    # SECTION 5: FINAL DATASET METRICS & SCHEMA
    # -------------------------------------------------------------
    add_heading_1("5. Final Research Dataset Schema & Verification Metrics")

    p = doc.add_paragraph()
    p.add_run(
        "The final verified dataset ("
    )
    p.add_run("political_verified.parquet").font.bold = True
    p.add_run(" / ")
    p.add_run("political_verified.csv").font.bold = True
    p.add_run(") contains 16,367 records across 18 standardized columns. The schema details are listed below:")

    schema_data = [
        ("Column Name", "Data Type", "Description & Operational Definition"),
        ("url", "String", "Cleaned canonical web URL of the news article."),
        ("website", "String", "Standardized news portal publisher name (e.g. Citizen, Star, Standard, TV47, Tuko, Nation)."),
        ("article_text", "String", "Full cleaned body text of the article extracted from HTML paragraphs."),
        ("political_classification", "String", "First-stage zero-shot political relevance label ('Political')."),
        ("bias_classification", "String", "Primary LLM (GPT-5.6-Luna) media framing bias annotation ('biased' or 'not_biased')."),
        ("bias_reason", "String", "GPT-5.6-Luna textual rationale justifying the framing bias classification."),
        ("judge_evidence_present", "Boolean", "Independent LLM Judge (Gemini 3.1) indicator if framing evidence exists."),
        ("judge_evidence_type", "List[String]", "Framing mechanisms detected (e.g. loaded_language, selective_framing, omission)."),
        ("judge_evidence_quote", "String", "Verbatim textual snippet quoted directly from the article text by Gemini Judge."),
        ("judge_verifier_reason", "String", "Detailed reasoning produced by Gemini Judge during independent stage-1 evaluation."),
        ("judge_independent_label", "String", "Independent bias label assigned by Gemini Judge ('biased' or 'not_biased')."),
        ("judge_initial_label_supported", "Boolean", "Judge assessment whether Luna's initial bias label is supported by text."),
        ("judge_initial_reason_supported", "Boolean", "Judge assessment whether Luna's reasoning logically matches article text."),
        ("judge_disagreement_reason", "String", "Explanation produced by Judge if Luna and Judge labels disagree."),
        ("judge_confidence", "String", "Judge confidence score for its assessment ('high', 'medium', 'low')."),
        ("luna_label_norm", "String", "Normalized lowercase version of Luna's bias label for metric calculation."),
        ("agree", "Boolean", "Boolean indicator whether Luna and Gemini Judge independent labels match."),
        ("quote_grounded", "Boolean", "Boolean verification confirming judge_evidence_quote appears verbatim in article_text."),
    ]

    tbl_schema = doc.add_table(rows=len(schema_data), cols=3)
    tbl_schema.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_schema.autofit = False
    set_table_borders(tbl_schema)
    s_widths = [Inches(2.2), Inches(1.1), Inches(3.2)]

    for r_i, r_c in enumerate(schema_data):
        row = tbl_schema.rows[r_i]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if r_i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        is_h = (r_i == 0)

        for c_i, val in enumerate(r_c):
            cell = row.cells[c_i]
            cell.width = s_widths[c_i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)

            if is_h:
                set_cell_background(cell, "1F4E78")
            else:
                set_cell_background(cell, "FFFFFF" if r_i % 2 == 1 else "F9FAFC")

            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = 'Calibri'
            if is_h:
                run.font.bold = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = WHITE
            else:
                run.font.size = Pt(8.5)
                run.font.color.rgb = DARK_TEXT

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(12)

    add_heading_2("Inter-Model Verification Metrics Summary")

    add_callout(
        "Inter-Annotator / Model Verification Statistics (N = 16,367):\n"
        "• Raw Inter-Model Agreement: 71.14% (95% Bootstrap CI: [70.47%, 71.85%])\n"
        "• Cohen's Kappa: 0.240 (indicates moderate chance-corrected agreement)\n"
        "• PABAK (Prevalence-Adjusted Bias-Adjusted Kappa): 0.423\n"
        "• Confirmed Biased Cases: 1,165 articles confirmed biased by both Luna and Gemini\n"
        "• Confirmed Neutral Cases: 10,479 articles confirmed not_biased by both models\n"
        "• Disagreement Cases Exported for Review: 4,723 articles in judge_disagreements.csv",
        "VERIFICATION AUDIT RESULTS"
    )

    doc.save("Kenyan_News_Pipeline_and_Scripts_Analysis.docx")
    print("Document successfully updated: Kenyan_News_Pipeline_and_Scripts_Analysis.docx")

if __name__ == "__main__":
    create_document()
