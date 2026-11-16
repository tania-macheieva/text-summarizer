"""Тести REST API: health, самаризація, валідація, файли, evaluation, train."""
import io
import json
import time

from docx import Document
from reportlab.pdfgen import canvas

from tests.conftest import SAMPLE_EN, SAMPLE_UK


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_summarize_success(client):
    r = client.post("/api/summarize", json={"text": SAMPLE_EN, "ratio": 0.4})
    assert r.status_code == 200
    d = r.json()
    for key in ("summary", "key_sentences", "original_word_count", "summary_word_count",
                "compression_ratio", "processing_time", "model_name"):
        assert key in d
    assert d["model_name"] == "random_forest" and d["used_fallback"] is False
    assert 0 < d["summary_word_count"] < d["original_word_count"]
    assert 0 < d["compression_ratio"] < 1
    assert d["key_sentences"] and all(k["text"] in d["summary"] for k in d["key_sentences"])


def test_summarize_keeps_original_order(client):
    d = client.post("/api/summarize", json={"text": SAMPLE_EN, "num_sentences": 3}).json()
    assert d["summary_sentence_count"] == 3 and len(d["key_sentences"]) == 3
    by_position = sorted(d["key_sentences"], key=lambda k: k["index"])
    # резюме = обрані речення, склеєні у ПОЧАТКОВОМУ порядку документа
    assert d["summary"] == " ".join(k["text"] for k in by_position)
    # key_sentences впорядковані за важливістю (rank 1, 2, 3)
    assert [k["rank"] for k in d["key_sentences"]] == sorted(k["rank"] for k in d["key_sentences"])


def test_summarize_ukrainian(client):
    r = client.post("/api/summarize", json={"text": SAMPLE_UK, "language": "auto", "num_sentences": 2})
    assert r.status_code == 200
    d = r.json()
    assert d["language"] == "uk" and d["summary_sentence_count"] == 2


def test_summarize_algorithms(client):
    for alg in ("random_forest", "tfidf", "lead"):
        r = client.post("/api/summarize", json={"text": SAMPLE_EN, "algorithm": alg, "num_sentences": 2})
        assert r.status_code == 200 and r.json()["summary_sentence_count"] == 2
    lead = client.post("/api/summarize", json={"text": SAMPLE_EN, "algorithm": "lead", "num_sentences": 2}).json()
    assert lead["summary"].startswith("Maternity pay for new mothers")


def test_empty_and_invalid_text(client):
    assert client.post("/api/summarize", json={"text": ""}).status_code == 422
    assert client.post("/api/summarize", json={"text": "   \n  "}).status_code == 422
    assert client.post("/api/summarize", json={"text": "!!! ??? ..."}).status_code == 422
    assert client.post("/api/summarize", json={}).status_code == 422
    assert client.post("/api/summarize", json={"text": 123}).status_code == 422


def test_parameter_validation(client):
    bad = [{"text": SAMPLE_EN, "ratio": 0}, {"text": SAMPLE_EN, "ratio": 1.5},
           {"text": SAMPLE_EN, "num_sentences": 0}, {"text": SAMPLE_EN, "language": "xx"},
           {"text": SAMPLE_EN, "algorithm": "gpt"}, {"text": SAMPLE_EN, "ratio": 0.3, "num_sentences": 2},
           {"text": "x" * 200_001}]
    for payload in bad:
        r = client.post("/api/summarize", json=payload)
        assert r.status_code == 422, payload
        assert "detail" in r.json()


def test_single_sentence_text(client):
    r = client.post("/api/summarize", json={"text": "This is just one short sentence about nothing."})
    assert r.status_code == 200 and r.json()["summary_sentence_count"] == 1


def test_file_txt(client):
    r = client.post("/api/summarize/file", files={"file": ("a.txt", SAMPLE_EN.encode(), "text/plain")},
                    data={"num_sentences": "2"})
    assert r.status_code == 200 and r.json()["filename"] == "a.txt"


def test_file_txt_cp1251_ukrainian(client):
    r = client.post("/api/summarize/file", files={"file": ("u.txt", SAMPLE_UK.encode("cp1251"), "text/plain")},
                    data={"language": "uk", "num_sentences": "2"})
    assert r.status_code == 200 and r.json()["language"] == "uk"


def test_file_docx(client):
    doc = Document()
    for para in SAMPLE_EN.split("\n\n"):
        doc.add_paragraph(para)
    buf = io.BytesIO(); doc.save(buf)
    r = client.post("/api/summarize/file", files={"file": ("a.docx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                    data={"ratio": "0.5"})
    assert r.status_code == 200 and r.json()["summary_sentence_count"] >= 1


def test_file_pdf(client):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for s in SAMPLE_EN.replace("\n\n", " ").split(". "):
        c.drawString(40, y, s[:110]); y -= 18
    c.save()
    r = client.post("/api/summarize/file", files={"file": ("a.pdf", buf.getvalue(), "application/pdf")})
    assert r.status_code == 200 and r.json()["original_word_count"] > 10


def test_file_errors(client):
    r = client.post("/api/summarize/file", files={"file": ("a.exe", b"abc", "application/octet-stream")})
    assert r.status_code == 415
    r = client.post("/api/summarize/file", files={"file": ("a.txt", b"", "text/plain")})
    assert r.status_code == 422
    r = client.post("/api/summarize/file", files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 422
    r = client.post("/api/summarize/file", files={"file": ("a.docx", b"not a docx", "application/octet-stream")})
    assert r.status_code == 422
    r = client.post("/api/summarize/file", files={"file": ("a.txt", b"hello world. second one.", "text/plain")},
                    data={"ratio": "3"})
    assert r.status_code == 422
    assert client.post("/api/summarize/file").status_code == 422


def test_models_endpoint(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    ids = {m["id"]: m for m in r.json()}
    assert set(ids) == {"random_forest", "tfidf", "lead"} and ids["random_forest"]["trained"] is True


def test_dataset_endpoint(client, tmp_path):
    assert client.get("/api/dataset").status_code == 404
    (tmp_path / "artifacts" / "dataset_info.json").write_text(json.dumps({"name": "x", "statistics": {"documents": 1}}))
    r = client.get("/api/dataset")
    assert r.status_code == 200 and r.json()["name"] == "x"


def test_evaluation_endpoint(client, tmp_path):
    assert client.get("/api/evaluation").status_code == 404
    payload = {"methods": {"random_forest": {"rouge1_f1": 0.5}}, "test_documents": 3}
    (tmp_path / "artifacts" / "evaluation.json").write_text(json.dumps(payload))
    r = client.get("/api/evaluation")
    assert r.status_code == 200 and r.json()["methods"]["random_forest"]["rouge1_f1"] == 0.5


def test_train_endpoint(client, tmp_path):
    r = client.post("/api/train", json={"dataset": "demo", "n_estimators": [30], "max_depth": [5], "min_samples_leaf": [2]})
    assert r.status_code == 202
    for _ in range(120):
        st = client.get("/api/train/status").json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert st["state"] == "done", st
    assert st["result"]["dataset"] == "demo" and (tmp_path / "model.joblib").exists()
    assert client.post("/api/summarize", json={"text": SAMPLE_EN}).status_code == 200


def test_train_bad_request(client):
    assert client.post("/api/train", json={"dataset": "nope"}).status_code == 422
    assert client.post("/api/train", json={"max_documents": 3}).status_code == 422
