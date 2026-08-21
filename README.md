# Day 21 — Fine-tuning LLMs · Lab (Track 3)

> **AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning & An Toàn**
> Đi kèm deck `day21-fine-tuning-llms-lora-qlora.tex` (90 trang · 20 mục).

**Một câu tóm tắt lab:** fine-tune một model mở bằng LoRA — rồi **chứng minh** nó thắng
được chính model đó khi đã được prompt tử tế. Nếu không chứng minh được, phát hiện ra
điều đó cũng được tính điểm đầy đủ.

---

## Hai câu hỏi lab bắt bạn trả lời

1. **Phần được tính loss có đúng là câu trả lời không?** (NB1 — chạy được trên CPU)
2. **Bản fine-tune có thắng base model đã prompt tử tế không — và bạn có phát hiện được
   nếu nó không thắng?** (NB2 đóng băng mốc, NB5 phán quyết)

Mọi thứ còn lại là chi tiết kỹ thuật phục vụ hai câu này.

---

## Chọn tier

| Tier | Phần cứng | Model | VRAM (bf16 LoRA) | Chạy được gì |
|---|---|---|---|---|
| `CPU` | không GPU | Qwen3.5-0.8B | — | **NB1 + toàn bộ test** |
| `LAPTOP` | GPU 8–12 GB | Qwen3.5-2B | ~5 GB | tất cả |
| **`T4`** *(mặc định)* | Colab Free T4 16 GB | Qwen3.5-4B | ~10 GB | tất cả |
| `BIGGPU` | L4 / A100 / 3090+ | Qwen3.5-9B | ~22 GB | tất cả, nhanh hơn |

Đổi tier = sửa `COMPUTE_TIER` trong `.env`. Chi tiết: **[HARDWARE-GUIDE.md](HARDWARE-GUIDE.md)**.

> **Không có GPU?** Vẫn làm được NB1 và toàn bộ test suite — đó là phần lab kiểm tra
> thứ quyết định kết quả nhiều nhất. Phần huấn luyện thì dùng Colab Free T4.

---

## Quick start

### Colab (khuyến nghị)

Mở **[`colab/Lab21_RUN_ALL.ipynb`](https://colab.research.google.com/github/hieutrungdao/Day21-Track3-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb)**
→ Runtime → Change runtime type → **T4 GPU** → chạy lần lượt ô 1 → 4.

> **Mỗi lần repo đổi, hãy mở LẠI tab (reload), đừng chỉ reconnect.** Colab đọc mã
> notebook từ GitHub đúng **một lần**, lúc URL được mở; reconnect, đổi runtime hay máy ảo
> mới đều không nạp lại. Ô Setup cũ vẫn chạy "xanh" và vẫn in đúng commit mới nhất — vì
> chính nó `git pull` repo — nhưng cài theo *danh sách gói cũ*. Lỗi sẽ nổ ~10 phút sau,
> bên trong `get_peft_model()`. Xem F-19 trong `SIMULATION-FINDINGS.md`.

### Máy cá nhân

```bash
git clone https://github.com/hieutrungdao/Day21-Track3-Finetuning-Lab.git
cd Day21-Track3-Finetuning-Lab
cp .env.example .env

# Không GPU — NB1 + test (đủ để bắt đầu, ~2 phút)
make setup-cpu && make smoke && make nb1

# Có GPU — cài torch cho CUDA của bạn TRƯỚC (xem đầu requirements.txt), rồi:
make setup && make smoke
make pipeline        # NB1 -> NB5, ~100-130 phút trên T4 (đo thật, xem docs/)
make verify          # cổng kiểm tra trước khi nộp
```

### Các lệnh `make`

```
make setup-cpu    Cài bản CPU (NB1 + test)
make setup        Cài đầy đủ (GPU)
make smoke        Import + data + unit test, không cần GPU
make nb1 .. nb6   Chạy từng notebook
make pipeline     CORE: NB1 -> NB5
make verify       Gatekeeper trước khi nộp
make clean        Xoá artefact sinh ra (giữ corpus gốc)
```

---

## Sáu notebook

| NB | Tên | Thời gian (T4, đo thật) | Cần GPU | Nội dung |
|---|---|---|---|---|
| **1** | `01_data_and_mask` | **~25 giây** | ✗ | chat template · **mask proof** · p95 → `max_length` · split seed 42 |
| **2** | `02_baselines` | **~17–23 ph** | ✓ | **đóng băng eval** + đo baseline (a) và (b) **trước khi train** |
| **3** | `03_train_correct` | **~15–25 ph** | ✓ | cấu hình vùng-không-hối-tiếc; in `layer_types` của chính model |
| **4** | `04_misconfig_autopsy` | **~45–60 ph** | ✓ | 3 run đối chứng cùng step: `attn_only` · `wrong_lr` · `qlora` |
| **5** | `05_evaluate_and_verdict` | **~21 ph** | ✓ | 4 nhóm · bảng 3 baseline · **cổng hồi quy** · chấm 3 run đối chứng |
| 6 | `06_merge_and_serve` | ~10 ph | ✓ | merge + assert không tụt điểm + hot-swap adapter *(tuỳ chọn)* |

> **Ngân sách thật: ~100–130 phút** cho core trên T4 free (lần đầu cộng thêm ~1,5 ph tải
> 9,32 GB trọng số). Đo thật 2026-08-20 với fp16 — xem `docs/MEASURED-T4-2026-08-20.md`.
> Đây là **khoảng**: cùng một cấu hình 30 step chạy 1456 s rồi 1021 s trên đúng mã đó,
> vì T4 free bị chia sẻ. Sinh văn bản chiếm phần lớn: tập eval được sinh **ba lần** —
> baseline (a), baseline (b), và bản fine-tune. Đó là cái giá của thiết kế ba-baseline,
> và nó đáng.
>
> **Hai cần gạt khi bạn bị bó thời gian.** Cả hai đều để **mặc định** khi nộp bài;
> `results/` ghi lại nếu bạn chạy chế độ rút gọn.
>
> ```bash
> EVAL_LIMIT=8 make pipeline     # ít mẫu eval hơn -> phần SINH ngắn lại
> EPOCHS=1     make pipeline     # nửa số step -> phần HUẤN LUYỆN ngắn lại
> ```
>
> Đo thật: `EPOCHS=1 EVAL_LIMIT=8` chạy NB1+NB2+NB3+NB5 hết **17 phút** (13s / 1,9 ph /
> 9,2 ph / 5,7 ph). Thêm NB4 thì cộng khoảng 25–30 ph nữa — ba run đối chứng vẫn phải
> train thật, `EVAL_LIMIT` không rút ngắn phần đó.
>
> `EPOCHS` áp cho **cả NB3 lẫn NB4** — không thể chỉnh một nửa. Đó là cố ý: ba run đối
> chứng chỉ có nghĩa khi chúng chạy đúng bằng số step của `correct`, và `make verify`
> đọc `runs.csv` để kiểm tra điều đó thật sự đã xảy ra.
>
> **Nếu NB4 bị đứt giữa chừng, đừng chạy lại từ đầu.** Adapter nào đã lưu thì được bỏ
> qua; `FORCE_RETRAIN=1` để train lại tất cả, `ONLY=qlora` để train lại đúng một run.

NB1–NB5 là **core**. NB6 là tuỳ chọn (có điểm thưởng).

`notebooks/*.py` là **bản gốc** (jupytext `py:percent`) — chạy trực tiếp bằng `python`
được, mở bằng Jupyter cũng được. `colab/*.ipynb` sinh ra từ chúng bằng `make colab`.

---

## Bài toán

**Ticket CSKH tiếng Việt → JSON triage 4 trường** (`intent`, `urgency`, `product`,
`sentiment`). Chọn bài này vì mọi nhóm điểm đều có thang **khách quan** — không cần
LLM judge, nên không có "điểm cho không":

| Nhóm | Đo bằng |
|---|---|
| **target** | độ chính xác từng trường so với nhãn |
| **regression** | 15 câu hỏi kiến thức/chỉ dẫn phổ thông — fine-tune **không được** làm hỏng |
| **format** | JSON parse được + đủ 4 khoá |
| **latency** | ms/mẫu, greedy decode |

### Đổi dataset của riêng bạn

Được khuyến khích — nhưng **chạy hết một lượt với corpus mặc định trước** để có mốc.
Khi đổi: thêm `data/CUSTOM_DATASET.md` mô tả nguồn, kích thước và cách khử nhiễm; nếu
không, `make verify` sẽ báo FAIL vì checksum tập eval đã đổi. (Đó là chủ ý: sửa tập eval
sau khi thấy kết quả sẽ làm hỏng toàn bộ phép so sánh.)

---

## Ghi chú kết quả thực nghiệm

### NB1 — Dữ liệu, chat template và loss mask

- Môi trường chạy: CPU, model `Qwen/Qwen3.5-0.8B`.
- Corpus gồm 250 mẫu; thống kê độ dài token: mean 93,1; p50 93; p95 98;
  p99 100; lớn nhất 101 token.
- Thống kê p95 gợi ý `max_length=256`. Lab vẫn giữ `max_length=512` theo cấu hình
  của tier CPU để có thêm khoảng an toàn cho chat template và các mẫu dài hơn, đồng
  thời không làm thay đổi cấu hình mặc định dùng để so sánh giữa các lần chạy.
- Dữ liệu được chia cố định với `seed=42`: 225 mẫu train và 25 mẫu validation.
- Các artefact NB1 đã được tạo và kiểm tra:
  `results/template_check.json`, `results/mask_proof.json`,
  `results/token_stats.json`, `data/split/train.jsonl` và
  `data/split/val.jsonl`.

## Nộp bài

Xem **[rubric.md](rubric.md)** — 100 điểm + tối đa 15 thưởng, ba lựa chọn định dạng nộp.
Chạy `make verify` trước khi nén file: nó kiểm tra artefact **và** kiểm tra rằng phép so
sánh bạn được chấm là một phép so sánh công bằng.

Điểm không nằm ở chỗ fine-tune của bạn thắng. Điểm nằm ở chỗ bạn **biết** nó có thắng hay không.
