"""Наскрізна перевірка UI та створення скриншотів (Playwright, реальний backend).

Запуск (backend на :8000, frontend на :3000):  python -m scripts.ui_capture
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from docx import Document
from playwright.sync_api import sync_playwright

from app import config
from app.data.loader import load_dataset

OUT = Path(__file__).resolve().parents[2] / "docs" / "screenshots"
UI = os.getenv("UI_CAPTURE_URL", "http://127.0.0.1:3000")
API = os.getenv("UI_CAPTURE_API", "http://127.0.0.1:8000")

UK_TEXT = (
    "Уряд України ухвалив нову програму підтримки малого та середнього бізнесу. Програма передбачає пільгові "
    "кредити для підприємців, які працюють у регіонах, найбільш постраждалих від війни. Міністр економіки заявив, "
    "що перші кошти будуть доступні вже з наступного місяця. За його словами, на програму буде спрямовано "
    "понад 10 мільярдів гривень з державного бюджету. Експерти вважають, що ініціатива допоможе створити "
    "десятки тисяч нових робочих місць. Водночас представники бізнес-асоціацій наголошують на надмірній "
    "складності процедури подання заявок. Опозиція критикує документ за недостатню прозорість розподілу коштів. "
    "Остаточне рішення щодо параметрів програми має ухвалити парламент до кінця року. У разі схвалення "
    "перші виплати отримають близько п'яти тисяч компаній."
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = config.BBC_DIR if config.BBC_DIR.exists() else config.DEMO_DIR
    df, _ = load_dataset(data_dir)
    if data_dir == config.BBC_DIR and (config.ARTIFACTS_DIR / "split.json").exists():
        test_ids = json.loads((config.ARTIFACTS_DIR / "split.json").read_text())["test"]
        df = df[df["doc_id"].isin(test_ids)]
    politics = df[df["category"] == "politics"]
    technology = df[df["category"] == "tech"]
    if politics.empty or technology.empty:
        raise RuntimeError(f"UI capture requires politics and tech examples in {data_dir}")
    art = politics.iloc[min(3, len(politics) - 1)]
    art2 = technology.iloc[min(5, len(technology) - 1)]
    run: dict = {
        "article_main": art["doc_id"],
        "article_file": art2["doc_id"],
        "capture_dataset": "bbc" if data_dir == config.BBC_DIR else "demo",
    }

    with tempfile.TemporaryDirectory(prefix="summarizer-ui-") as temp_dir:
        tmp = Path(temp_dir)
        d = Document()
        for para in art2["text"].split("\n\n"):
            d.add_paragraph(para)
        d.save(tmp / "tech_article.docx")
        (tmp / "data.csv").write_text("a,b\n1,2", encoding="utf-8")

        with sync_playwright() as p:
            b = p.chromium.launch()
            ctx = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5, accept_downloads=True)
            ctx.grant_permissions(["clipboard-read", "clipboard-write"])
            page = ctx.new_page()
            msgs: list[str] = []
            page.on("console", lambda m: msgs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)

            # 1. Empty state
            page.goto(UI); page.wait_for_selector("text=Source text")
            page.wait_for_function("document.querySelectorAll('select')[1].options.length >= 3")
            page.screenshot(path=OUT / "01_main_empty.png")

            # 2. English article, reduction mode
            page.fill("textarea", art["text"])
            with page.expect_response(lambda r: r.url.endswith("/api/summarize") and r.request.method == "POST") as ri:
                page.click("button.primary")
            resp = ri.value
            run["en_status"] = resp.status; run["en_response"] = resp.json()
            page.wait_for_selector("[data-testid=summary]")
            shown = page.inner_text("[data-testid=summary]")
            assert shown == run["en_response"]["summary"], "UI output does not match the API response"
            page.screenshot(path=OUT / "02_result_en.png", full_page=True)

            # Copy / Download
            page.click("text=Copy"); page.wait_for_selector("text=Copied")
            run["clipboard_equals_summary"] = page.evaluate("navigator.clipboard.readText()") == shown
            with page.expect_download() as di:
                page.click("text=Download .txt")
            dl = di.value; dl.save_as(tmp / "summary.txt")
            run["download_equals_summary"] = (tmp / "summary.txt").read_text(encoding="utf-8") == shown
            run["download_name"] = dl.suggested_filename

            # 3. Sentence count and Lead-N
            page.click("text=Sentence count")
            page.fill("input[type=number]", "2")
            page.select_option("select >> nth=1", "lead")
            with page.expect_response(lambda r: r.url.endswith("/api/summarize")) as ri:
                page.click("button.primary")
            run["lead_response"] = ri.value.json()
            page.wait_for_selector("[data-testid=summary]")
            page.screenshot(path=OUT / "03_sentences_mode_lead.png", full_page=True)

            # 4. Ukrainian document
            page.select_option("select >> nth=1", "random_forest")
            page.click("text=Reduction")
            page.fill("textarea", UK_TEXT)
            with page.expect_response(lambda r: r.url.endswith("/api/summarize")) as ri:
                page.click("button.primary")
            run["uk_response"] = ri.value.json()
            page.wait_for_selector("[data-testid=summary]")
            page.screenshot(path=OUT / "04_result_uk.png", full_page=True)

            # 5. DOCX upload
            page.reload(); page.wait_for_selector("text=Source text")
            page.set_input_files("input[type=file]", str(tmp / "tech_article.docx"))
            with page.expect_response(lambda r: r.url.endswith("/api/summarize/file")) as ri:
                page.click("button.primary")
            resp = ri.value
            run["file_status"] = resp.status; run["file_response"] = resp.json()
            page.wait_for_selector("[data-testid=summary]")
            page.screenshot(path=OUT / "05_file_upload_docx.png", full_page=True)

            # 6. Unsupported upload and empty-input validation
            page.reload(); page.wait_for_selector("text=Source text")
            page.set_input_files("input[type=file]", str(tmp / "data.csv"))
            with page.expect_response(lambda r: r.url.endswith("/api/summarize/file")) as ri:
                page.click("button.primary")
            run["bad_file_status"] = ri.value.status
            page.wait_for_selector("[role=alert]")
            run["bad_file_message"] = page.inner_text("[role=alert]")
            page.screenshot(path=OUT / "06_error_unsupported_file.png")
            page.reload(); page.wait_for_selector("text=Source text")
            page.click("button.primary")
            page.wait_for_selector("[role=alert]")
            run["empty_message"] = page.inner_text("[role=alert]")

            # 7. Evaluation, About, Swagger
            page.goto(UI + "/evaluation"); page.wait_for_selector("table"); page.wait_for_selector(".recharts-bar-rectangle")
            page.wait_for_timeout(800)
            page.screenshot(path=OUT / "07_evaluation.png", full_page=True)
            page.goto(UI + "/about"); page.wait_for_selector("text=Dataset"); page.wait_for_timeout(500)
            page.screenshot(path=OUT / "08_about.png", full_page=True)
            page.goto(API + "/docs"); page.wait_for_selector(".opblock-summary"); page.wait_for_timeout(500)
            page.screenshot(path=OUT / "09_swagger.png", full_page=True)
            run["console_errors"] = [m for m in msgs if "favicon" not in m]
            b.close()
    (OUT / "ui_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in run.items() if not k.endswith("_response")}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for k in ("en_response", "lead_response", "uk_response", "file_response"):
        r = run[k]; print(k, r["model_name"], r["language"], r["original_word_count"], "->", r["summary_word_count"],
                          r["compression_ratio"], r["processing_time"], r["warnings"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
