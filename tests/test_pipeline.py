import csv, json
from pathlib import Path
import pytest
from exam_prep import config, export, extract, review, split, syllabus, tag
from exam_prep.cli import main
from exam_prep.llm import GeminiLLM, LLMError, _cache_key, load_prompt
from exam_prep.metadata import discover_papers, parse_filename
from exam_prep.schema import LLMTagBatch, PageText, PaperMeta, Question, TopicTag
from exam_prep.store import read_jsonl


# ---------- metadata ----------

def test_filename_parsing():
    m = parse_filename(Path("dbms_2024_9_mid.pdf"))
    assert (m.paper_id, m.subject, m.year, m.month, m.exam_type) == ("DBMS_2024_09_MID", "DBMS", 2024, 9, "mid")
    assert parse_filename(Path("OS_2023_05_REM.pdf")).exam_type == "remedial"
    assert parse_filename(Path("scan0001.pdf")) is None


def test_manifest_overrides_bad_names(workspace):
    (config.RAW_DIR / "OS" / "scan0001.pdf").write_bytes((config.RAW_DIR / "OS" / "OS_2025_11_END.pdf").read_bytes())
    assert [m.paper_id for m in discover_papers()] == ["OS_2025_11_END"]          # unnamed file skipped
    config.MANIFEST.write_text("file,subject,year,month,exam_type\nscan0001.pdf,os,2024,5,end\n", encoding="utf-8")
    assert [m.paper_id for m in discover_papers()] == ["OS_2024_05_END", "OS_2025_11_END"]


def test_external_papers_keep_their_source(workspace):
    ext = config.EXTERNAL_DIR / "OS"
    ext.mkdir(parents=True)
    src = (config.RAW_DIR / "OS" / "OS_2025_11_END.pdf").read_bytes()
    (ext / "BITM-OS_2019_11_END.pdf").write_bytes(src)
    (ext / "OS_2019_05_END.pdf").write_bytes(src)                   # no source prefix: rejected
    metas = {m.paper_id: m for m in discover_papers()}
    assert set(metas) == {"OS_2025_11_END", "BITM-OS_2019_11_END"}
    assert metas["BITM-OS_2019_11_END"].source == "BITM" and metas["OS_2025_11_END"].source == "DDU"
    assert [m.paper_id for m in discover_papers(external=False)] == ["OS_2025_11_END"]
    assert [m.paper_id for m in discover_papers(source="external")] == ["BITM-OS_2019_11_END"]
    assert [m.paper_id for m in discover_papers(source="DDU")] == ["OS_2025_11_END"]


def test_hex_watermark_removed():
    assert extract.clean_text_layer("DA7E51154F8C49B79E87EA53807396A0 Q.1 Explain RPC. [5]") == "Q.1 Explain RPC. [5]"


# ---------- extract ----------

def test_extract_uses_text_layer_then_vision_for_scans(workspace, fake_llm):
    meta = discover_papers()[0]
    pages = extract.extract_paper(meta, fake_llm)
    assert [p.method for p in pages] == ["text", "ocr"]
    assert "Banker's algorithm" in pages[0].text
    assert fake_llm.calls[0][0] == "LLMPage" and fake_llm.calls[0][2] == 1      # one page image sent


def test_scanned_page_without_key_fails_clearly(workspace):
    with pytest.raises(RuntimeError, match="looks scanned"):
        extract.extract_paper(discover_papers()[0], None)


# ---------- split ----------

@pytest.fixture
def extracted(workspace, fake_llm):
    metas = discover_papers()
    extract.run(metas, fake_llm)
    return metas


def test_rules_split(extracted):
    qs = split.split_rules(extracted[0], extract.load_pages("OS_2025_11_END"))
    ids = [q.question_id for q in qs]
    assert ids == ["OS_2025_11_END-Q1a", "OS_2025_11_END-Q1b", "OS_2025_11_END-Q1c", "OS_2025_11_END-Q1d", "OS_2025_11_END-Q1e",
                   "OS_2025_11_END-Q2a", "OS_2025_11_END-Q2b", "OS_2025_11_END-Q2b-OR",
                   "OS_2025_11_END-Q3", "OS_2025_11_END-Q3-OR", "OS_2025_11_END-Q4a", "OS_2025_11_END-Q4b"]
    by_id = {q.question_id: q for q in qs}
    assert all(q.marks is not None for q in qs)
    assert by_id["OS_2025_11_END-Q1a"].marks == 2 and by_id["OS_2025_11_END-Q2a"].marks == 6
    assert "P1 0 8" in by_id["OS_2025_11_END-Q2b"].text            # data lines stay with their question
    assert by_id["OS_2025_11_END-Q2b-OR"].is_or_alternative and by_id["OS_2025_11_END-Q3-OR"].is_or_alternative
    assert not by_id["OS_2025_11_END-Q2b"].is_or_alternative
    assert "Attempt any" not in by_id["OS_2025_11_END-Q1a"].text      # instruction stem dropped
    assert by_id["OS_2025_11_END-Q4b"].page == 2
    assert not any("Page" in q.text or "DHARMSINH" in q.text for q in qs)
    assert split.rules_look_ok(qs)


DDU_PAGE = """DHARMSINH DESAI UNIVERSITY, NADIAD
Q.1     Do as directed.                                                     [12]
CO1  R  (a)  1.​ Which instruction exposes a port?                       [1]
                   (i) ENV      (ii) EXPOSE
             2.​ What is the default docker network?                       [1]
CO2  U  (b)  How does Hadoop ensure fault tolerance?                           [2]
Q.2     Do as directed                                                      [12]
CO3 A    (a)  Write a docker-compose.yml file that includes:                 [06]
                    (a) A container for the backend
                    (b) A container for the database
CO5  E   (b)    PID  P_Name  P_Cost                                [06]               CID  Cat_Name
               Write LINQ queries for the relations above.
                            OR
CO3 A    (a)  Apply the concept of image map.                                 [06]
CO6 C    (b)  Create an upload form.                                          [06]
Q.3      Attempt all of the following questions.                             [12]
CO6 C   (a)  Explain XAI.                                                      [12]
Q.3      Attempt all of the following questions.                             [12]
CO6(C)  (a)  Explain LSTM gates.                                               [12]
             Draw the cell diagram.                       Page 1 of 2
                                Page 1 of 1"""


def test_rules_split_ddu_layout():
    meta = PaperMeta(paper_id="DC_2025_09_MID", subject="DC", year=2025, month=9, exam_type="mid", source_file="x.pdf")
    qs = {q.question_id: q for q in split.split_rules(meta, [PageText(paper_id=meta.paper_id, page=1, method="text", text=DDU_PAGE)])}
    assert list(qs) == ["DC_2025_09_MID-Q1a", "DC_2025_09_MID-Q1b", "DC_2025_09_MID-Q2a", "DC_2025_09_MID-Q2b",
                        "DC_2025_09_MID-Q2a-OR", "DC_2025_09_MID-Q2b-OR", "DC_2025_09_MID-Q3a", "DC_2025_09_MID-Q3a-OR"]
    q1a = qs["DC_2025_09_MID-Q1a"]
    assert (q1a.co, q1a.bloom, q1a.marks) == ("CO1", "R", 2)                  # CO prefix parsed, [1]+[1] summed
    assert "​" not in q1a.text and "Do as directed" not in q1a.text
    assert "(b) A container for the database" in qs["DC_2025_09_MID-Q2a"].text   # nested list stays inside its sub-part
    assert qs["DC_2025_09_MID-Q2b"].marks == 6 and "LINQ" in qs["DC_2025_09_MID-Q2b"].text   # marks mid-line, table to the right
    assert qs["DC_2025_09_MID-Q2b-OR"].is_or_alternative                     # labels restarted after OR
    assert qs["DC_2025_09_MID-Q3a-OR"].is_or_alternative and qs["DC_2025_09_MID-Q3a-OR"].bloom == "C"   # repeated Q.3, no OR line
    assert qs["DC_2025_09_MID-Q3a-OR"].text == "Explain LSTM gates. Draw the cell diagram."              # footer on a text line dropped


def test_llm_split_flags_text_not_in_paper(extracted, fake_llm):
    qs, unverified = split.split_llm(extracted[0], extract.load_pages("OS_2025_11_END"), fake_llm)
    assert len(qs) == 3 and unverified == 1
    assert qs[0].page == 1 and qs[2].page is None


def test_auto_falls_back_to_llm_on_unreadable_layout(workspace, fake_llm):
    meta = PaperMeta(paper_id="X_2025_11_END", subject="X", year=2025, month=11, source_file="x.pdf")
    pages = [PageText(paper_id=meta.paper_id, page=1, method="text", text="1 Define process 2 Explain Banker's algorithm")]
    from exam_prep.store import write_jsonl
    write_jsonl(extract.pages_path(meta.paper_id), pages)
    qs = split.split_paper(meta, "auto", fake_llm)
    assert qs and qs[0].split_method == "llm"


def test_split_run_merges_by_paper(extracted):
    split.run(extracted, "rules")
    split.run(extracted, "rules")                     # re-running must not duplicate rows
    assert len(read_jsonl(config.QUESTIONS_FILE, Question)) == 12


# ---------- syllabus + tag ----------

@pytest.fixture
def tagged(extracted, fake_llm):
    (config.SYLLABUS_DIR).mkdir(parents=True, exist_ok=True)
    (config.SYLLABUS_DIR / "OS.txt").write_text("Unit 1 Processes ... Unit 4 Memory management", encoding="utf-8")
    syllabus.run("OS", fake_llm)
    split.run(extracted, "rules")
    tag.run("OS", fake_llm)
    return fake_llm


def test_taxonomy_ids(tagged):
    tax = syllabus.load_taxonomy("OS")
    assert tax.subject_name == "Operating Systems"
    assert [t.topic_id for t in tax.topics][:3] == ["OS-U1-T1", "OS-U1-T2", "OS-U1-T3"]
    assert "OS-U4-T4" in tax.ids()


def test_tags_are_validated_and_batched(tagged):
    tags = {t.question_id: t for t in read_jsonl(config.TAGS_FILE, TopicTag)}
    assert len(tags) == 12
    assert tags["OS_2025_11_END-Q2a"].topic_ids == ["OS-U3-T3"]
    assert tags["OS_2025_11_END-Q4b"].topic_ids == ["OS-U4-T3"]
    bad = tags["OS_2025_11_END-Q1d"]                 # model returned an ID not in the taxonomy
    assert bad.topic_ids == [] and bad.needs_review
    assert tags["OS_2025_11_END-Q2a"].prompt_version == load_prompt("tag")[1]
    tag_calls = [c for c in tagged.calls if c[0] == "LLMTagBatch"]
    assert len(tag_calls) == 2 + 1                     # 12 questions in batches of 8, plus one solo retry


def test_retag_skips_done_questions(tagged):
    before = len(tagged.calls)
    tag.run("OS", tagged)
    assert len(tagged.calls) == before


# ---------- review + export ----------

def test_review_round_trip(tagged):
    path = review.sample(n=6, seed=1)
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert any(r["question_id"] == "OS_2025_11_END-Q1d" for r in rows)       # low-confidence/failed tags always sampled
    for r in rows:
        if r["question_id"] == "OS_2025_11_END-Q1d":
            r["verdict"], r["correct_topic_ids"] = "n", "OS-U4-T4"
        else:
            r["verdict"] = "y"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=review.FIELDS)
        w.writeheader()
        w.writerows(rows)
    rep = review.score()
    assert rep["judged"] == 6 and rep["strict_accuracy"] == round(5 / 6, 3)
    assert (config.REVIEW_DIR / f"report_{rep['version']}.md").exists()

    export.export()
    rows = {json.loads(l)["question_id"]: json.loads(l) for l in config.EXPORT_FILE.read_text(encoding="utf-8").splitlines()}
    fixed = rows["OS_2025_11_END-Q1d"]
    assert fixed["topic_ids"] == ["OS-U4-T4"] and fixed["topics"] == ["Thrashing"] and fixed["tag_source"] == "human"
    assert rows["OS_2025_11_END-Q2b"]["units"] == [2]

    s = export.stats()
    assert s["ddu_papers"] == 1 and s["external_questions"] == 0 and s["pages_via_gemini_vision"] == 1 and s["questions_tagged"] == 12 and s["hand_corrected_tags"] == 1


# ---------- LLM wrapper + CLI ----------

def test_cache_key_depends_on_prompt_model_and_images():
    k = _cache_key("m", "p", LLMTagBatch, None)
    assert k == _cache_key("m", "p", LLMTagBatch, None)
    assert k != _cache_key("m2", "p", LLMTagBatch, None)
    assert k != _cache_key("m", "p", LLMTagBatch, [b"img"])


def test_missing_key_message(workspace):
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM()


def test_cli_stats_and_errors(workspace, capsys):
    assert main(["stats"]) == 0
    assert main(["tag", "--subject", "OS"]) == 1                       # no key: clean error, no traceback
    assert "GEMINI_API_KEY" in capsys.readouterr().out


def test_off_syllabus_tags():
    from exam_prep.schema import LLMTag, Taxonomy, Topic

    class StubLLM:
        model = "stub"
        def generate_json(self, prompt, schema, images=None):
            return LLMTagBatch(tags=[LLMTag(question_id="BITM-DC_2019_11_END-Q2a", topic_ids=[], confidence="high", off_syllabus=True),
                                     LLMTag(question_id="DC_2025_09_MID-Q1a", topic_ids=[], confidence="low", off_syllabus=True),
                                     LLMTag(question_id="DC_2025_09_MID-Q1b", topic_ids=["DC-U9-T9"], confidence="high")])   # invented ID

    tax = Taxonomy(subject="DC", topics=[Topic(topic_id="DC-U4-T3", unit_no=4, unit_title="Docker", name="Dockerfile")])
    mk = lambda qid, src: Question(question_id=qid, paper_id=qid.rsplit("-Q", 1)[0], subject="DC", source=src, exam_type="end", year=2025, question_no="1", text="x")
    qs = [mk("BITM-DC_2019_11_END-Q2a", "BITM"), mk("DC_2025_09_MID-Q1a", "DDU"), mk("DC_2025_09_MID-Q1b", "DDU")]
    tags = {t.question_id: t for t in tag.tag_batch(qs, tax, StubLLM(), "$questions", "tag_vX")}
    ext, ddu = tags["BITM-DC_2019_11_END-Q2a"], tags["DC_2025_09_MID-Q1a"]
    assert ext.off_syllabus and not ext.needs_review and ext.topic_ids == []      # expected for another university
    assert ddu.off_syllabus and ddu.needs_review                                   # DDU question outside DDU syllabus: taxonomy gap
    assert "DC_2025_09_MID-Q1b" not in tags                                        # invented topic ID still rejected


def test_ai_review_is_not_counted_as_human(tagged):
    path = review.sample(n=6, seed=1)
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["verdict"], r["correct_topic_ids"] = ("n", "OS-U4-T4") if r["question_id"] == "OS_2025_11_END-Q1d" else ("y", "")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=review.FIELDS)
        w.writeheader()
        w.writerows(rows)
    rep = review.score(reviewer="claude")
    assert rep["reviewer"] == "claude"
    assert "not a human hand check" in (config.REVIEW_DIR / f"report_{rep['version']}.md").read_text(encoding="utf-8")
    assert tag.load_overrides()["OS_2025_11_END-Q1d"].source == "claude"
    s = export.stats()
    assert s["hand_corrected_tags"] == 0 and s["ai_review_corrected_tags"] == 1
