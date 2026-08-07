# Day 11 — Controlled Agent Security (2026)

**Họ tên:** Nguyễn Thừa Tuân · **MSSV:** 2A202601330

Làm sao để ứng dụng agent an toàn hơn?

**Hình thức:** bài tập **cá nhân** (1 người / 1 MSSV).

**Đề bài duy nhất:** [`assignment11.md`](assignment11.md) · Cách nộp: [`SUBMISSION.md`](SUBMISSION.md)

---

## Cài đặt môi trường (làm trước)

```powershell
# 1) Tạo + kích hoạt virtualenv (khuyến nghị)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) API key
Copy-Item .env.example .env
# Mở .env, dán GOOGLE_API_KEY — lấy tại https://aistudio.google.com/apikey
# và OPENAI_API_KEY — lấy tại https://platform.openai.com/api-keys

# 3) Cài dependency trong venv
python -m pip install -U pip
pip install -r requirements.txt
```

> **Lưu ý model:** `src/agents/agent.py`, `output_guardrails.py`, `nemo_guardrails.py`, `attacks.py` được route qua OpenAI `gpt-4o-mini` (qua `google.adk.models.lite_llm.LiteLlm`) vì Gemini free-tier trả `429 RESOURCE_EXHAUSTED` liên tục trong ngày làm bài (chi tiết + lý do trong `report/2A202601330_report.md`) — cần cả `GOOGLE_API_KEY` (giá trị bất kỳ, không còn gọi thật) lẫn `OPENAI_API_KEY` (giá trị thật) mới chạy được. Riêng `src/agents/guards_agent.py` (mục tiêu bonus, file tham khảo) **vẫn giữ nguyên Gemini** như bản gốc.

Mỗi lần mở terminal mới: `.\.venv\Scripts\Activate.ps1` rồi mới chạy code.

Nếu PowerShell báo không cho chạy script:  
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

PowerShell (nếu chưa load `.env`):

```powershell
$env:GOOGLE_API_KEY="dummy-hoặc-key-thật"
$env:OPENAI_API_KEY="key-openai-thật"
```

---

## Rubric (tóm tắt)

Chi tiết đầy đủ trong [`assignment11.md`](assignment11.md).

| Năng lực | Điểm | Bạn làm gì |
|---|---:|---|
| Direct + indirect guardrails | 35 | Xử lý jailbreak, email/RAG untrusted, Unicode và false positive |
| Permission + HITL | 35 | Egress allowlist, high-risk action, approval/reject/timeout/audit |
| Output + incident response | 20 | Redact PII/secret, monitoring, correlation trace |
| Red team | 10 | Attack taxonomy và report source-to-sink |
| Bonus | +10 | Verifier replay xác nhận Guards leak; không tin transcript tự khai |

**Gợi ý:** làm **Phòng thủ (A)** trước, **Tấn công (B)** sau.

**Hạn nộp:** Thứ sáu **7/8**, **23:59 giờ Việt Nam (ICT)**.

| Tài liệu | Dùng để |
|----------|---------|
| [`assignment11.md`](assignment11.md) | **Đề bài duy nhất** (rubric + cách chạy A/B) |
| [`SUBMISSION.md`](SUBMISSION.md) | Cách nộp, tên file, cấu trúc thư mục |

---

## Timeline buổi lab

Hình thức: **cá nhân** (1 người / 1 MSSV). Luồng: **Setup → A → Break → B → Break → Demo**.

| # | Phần | Nội dung | Thời lượng |
|---|------|----------|-----------:|
| 0 | **Setup** | Cài đặt môi trường (`pip`, `GOOGLE_API_KEY`, chạy local) | 30' |
| 1 | **A · Phòng thủ** | 2A Input · 2B Output · 2C NeMo · Part 3 Testing · Part 4 HITL | 120' |
| — | **Break** | Nghỉ giải lao | 10' |
| 2 | **B · Tấn công** | Tấn công **Unsafe** (điểm B) + **Guards** (điểm cộng nếu LEAKED) | 60' |
| — | **Break** | Nghỉ giải lao | 10' |
| 3 | **Demo** | Demo cá nhân · attack prompting (cuối buổi) | 45' |
| | **Tổng** | Nội dung lab (+ nghỉ) | **245'** |
| | | + Setup | **+30'** |

**Điểm cộng Demo (trên lớp):** lên demo **+1** (nếu defense chặn thành công ≥5 prompt thì **×2**) · tấn công thành công (LEAKED) **+2**.

Chi tiết mốc Part B (60'):

| Mốc | Việc làm |
|-----|----------|
| 0–25' | TODO 13 — viết ≥5 prompt tấn công nâng cao trong `src/attacks/attacks.py` |
| 25–45' | Chạy attack trên unsafe rồi guards; quan sát `LEAKED` / `no secret leak` |
| 45–60' | TODO 14 — AI red team ≥5 attack; lưu `outputs/attack_results.json` |

Slide đầy đủ + timer trên lớp: [`Slide_Lab_Day11.html`](Slide_Lab_Day11.html).

---

## Tình huống

Chatbot ngân hàng **VinBank**. Agent “unsafe” cố ý chứa mật khẩu / API key trong system prompt.

```
Câu hỏi người dùng
    → Rate Limiter
    → Lọc đầu vào (Input Guardrails)
    → LLM trả lời
    → Lọc đầu ra (Output Guardrails + Judge)
    → Audit / Monitoring
    → Phản hồi
```

---

## Làm bài trên máy

> Đã cài môi trường ở mục **Cài đặt môi trường** phía trên chưa? Nếu chưa thì làm trước.

### Phần A — Phòng thủ

**Thứ tự:** sửa TODO trong file → rồi mới chạy lệnh. Chi tiết: [`assignment11.md`](assignment11.md) §5.

| Làm trước | File |
|-----------|------|
| TODO **1–3** | `src/guardrails/input_guardrails.py` |
| TODO **4–6** | `src/guardrails/output_guardrails.py` |
| TODO **7** (tuỳ chọn) | `src/guardrails/nemo_guardrails.py` |
| TODO **8** (+ egress 8A) | `src/assignment/*.py` → rồi `python main.py --part 5` |
| TODO **9–10** | `src/testing/testing.py` |
| TODO **11–12** | `src/hitl/hitl.py` |
| TODO **13–14** (phần B) | `src/attacks/attacks.py` |

Sau khi đã code, kiểm:

```powershell
cd src
python main.py --part 2    # sau TODO 1–6 (+7 NeMo)
python main.py --part 3    # sau TODO 9–10
python main.py --part 4    # sau TODO 11–12
python main.py --part 5    # sau TODO 8 → outputs/results.json (+ audit/metrics)
```

```powershell
pytest tests/smoke -q
pytest tests/public -q
python scripts/grade.py --submission-dir . --out outputs/grade_report.json
```

Viết `report/<MSSV>_report.md`.

### Phần B — Red team và bonus

1. Viết ≥5 prompt vào `src/attacks/attacks.py`
2. Chạy (tấn công **unsafe** rồi **guards**):

```powershell
cd src
python main.py --part 1
```

3. Unsafe = attack target để phân tích. Guards (`src/agents/guards_agent.py`) = **bonus chỉ khi verifier replay xác nhận leak**.
4. Lưu `outputs/attack_results.json` làm evidence; không tự cấp runtime score hoặc bonus.

Colab / Jupyter (tuỳ chọn): `notebooks/lab11_guardrails_hitl.ipynb`. Local là đủ.

Nộp theo [`SUBMISSION.md`](SUBMISSION.md).

---

## Cấu trúc repo

```
├── assignment11.md                    ← Đề bài duy nhất
├── SUBMISSION.md                      ← Quy định nộp
├── data/pii_hallucination_samples.json ← PII + ground_truth đối chiếu hallucination
├── src/
│   ├── assignment/                    ← Hạng mục A (Phòng thủ) — starters
│   ├── attacks/                       ← Hạng mục B (Tấn công)
│   ├── agents/security_boundary.py    ← Reference provenance / action boundary
│   ├── agents/guards_agent.py         ← Guards Agent (mục tiêu bonus)
│   ├── guardrails/ testing/ hitl/     ← Module hỗ trợ phòng thủ
│   └── main.py
├── notebooks/lab11_guardrails_hitl.ipynb
├── schemas/results.schema.json
├── scripts/grade.py
├── tests/
├── Slide_Lab_Day11.html
└── .env.example
```

---

## Tài liệu tham khảo

- [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [Google ADK](https://google.github.io/adk-docs/)
- [AI Safety Fundamentals](https://aisafetyfundamentals.com/)
