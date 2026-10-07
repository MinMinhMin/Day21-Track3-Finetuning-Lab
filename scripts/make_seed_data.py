#!/usr/bin/env python3
"""Generate the lab's committed seed corpus — deterministic, offline, Vietnamese.

Task: **e-commerce support ticket -> structured triage JSON**.

Chosen deliberately over a free-form chat task because every evaluation group then
has an *objective* scorer:
  * target      -> per-field accuracy against a known label (no LLM judge, no free points)
  * format      -> does it emit parseable JSON with the required keys
  * regression  -> a general-knowledge/instruction slice the fine-tune must not break

A base model with a naive prompt does badly at this; a base model with a *good* prompt
does respectably. That is the point: baseline (b) has to be a real bar to clear
(deck §21), otherwise "my fine-tune won" means nothing.

Run:  python scripts/make_seed_data.py
"""
from __future__ import annotations

import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEED = 20260821

INTENTS = {
    "doi_tra": ["muốn đổi", "cho tôi trả lại", "đổi size", "trả hàng", "hoàn lại"],
    "van_chuyen": ["giao hàng chậm", "chưa nhận được", "đơn đang ở đâu", "shipper không gọi"],
    "hoan_tien": ["hoàn tiền", "trả lại tiền", "khi nào có tiền về", "chưa thấy tiền"],
    "san_pham_loi": ["bị lỗi", "vỡ khi nhận", "không hoạt động", "sai màu", "thiếu phụ kiện"],
    "hoi_thong_tin": ["còn hàng không", "bảo hành bao lâu", "có màu khác không", "giá bao nhiêu"],
}
URGENCY_MARKERS = {
    "cao": ["gấp", "ngay lập tức", "quá hạn rồi", "tôi cần trước ngày mai", "khẩn"],
    "trung_binh": ["sớm nhé", "mong shop phản hồi", "đã 3 ngày rồi"],
    "thap": ["khi nào tiện", "không vội", "hỏi cho biết thôi"],
}
SENTIMENTS = {
    "tieu_cuc": ["rất thất vọng", "quá tệ", "bực mình", "lần cuối mua ở đây"],
    "trung_tinh": ["nhờ shop kiểm tra", "cho tôi hỏi", "shop xem giúp"],
    "tich_cuc": ["shop hỗ trợ tốt", "cảm ơn shop nhiều", "mình vẫn tin tưởng shop"],
}
PRODUCTS = [
    "tai nghe bluetooth", "áo khoác gió", "nồi chiên không dầu", "bàn phím cơ",
    "giày chạy bộ", "sạc dự phòng", "máy xay sinh tố", "balo laptop",
    "đèn bàn LED", "chuột không dây", "bình giữ nhiệt", "ốp lưng điện thoại",
]
OPENERS = ["Chào shop,", "Shop ơi,", "Alo shop,", "Xin chào,", "Cho mình hỏi,"]
ORDER_PREFIX = ["DH", "OD", "VN"]

REQUIRED_KEYS = ["intent", "urgency", "product", "sentiment"]


def make_ticket(rng: random.Random) -> dict:
    intent = rng.choice(list(INTENTS))
    urgency = rng.choice(list(URGENCY_MARKERS))
    sentiment = rng.choice(list(SENTIMENTS))
    product = rng.choice(PRODUCTS)
    order = f"{rng.choice(ORDER_PREFIX)}{rng.randint(100000, 999999)}"

    body = (
        f"{rng.choice(OPENERS)} mình đặt {product} mã đơn {order}. "
        f"{rng.choice(INTENTS[intent]).capitalize()}. "
        f"{rng.choice(URGENCY_MARKERS[urgency]).capitalize()}. "
        f"{rng.choice(SENTIMENTS[sentiment]).capitalize()}."
    )
    label = {"intent": intent, "urgency": urgency, "product": product, "sentiment": sentiment}
    return {"ticket": body, "label": label, "order_id": order}


INSTRUCTION = (
    "Phân loại ticket chăm sóc khách hàng sau thành JSON với đúng 4 khóa: "
    "intent, urgency, product, sentiment. Chỉ trả về JSON, không giải thích.\n\n"
    "intent thuộc: doi_tra | van_chuyen | hoan_tien | san_pham_loi | hoi_thong_tin\n"
    "urgency thuộc: cao | trung_binh | thap\n"
    "sentiment thuộc: tieu_cuc | trung_tinh | tich_cuc\n"
    "product: tên sản phẩm xuất hiện trong ticket."
)


def to_record(t: dict) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": t["ticket"],
        "output": json.dumps(t["label"], ensure_ascii=False),
        "label": t["label"],
    }


# A general-capability slice: short VN instruction-following items unrelated to triage.
# If the fine-tune starts answering these in triage-JSON, that is catastrophic
# forgetting (deck §6.3) and the regression group is what catches it.
REGRESSION_ITEMS = [
    ("Thủ đô của Việt Nam là thành phố nào?", ["Hà Nội"]),
    ("Kể tên hai đại dương lớn nhất thế giới.", ["Thái Bình Dương", "Đại Tây Dương"]),
    ("1 km bằng bao nhiêu mét?", ["1000"]),
    ("Viết một câu chúc mừng sinh nhật bằng tiếng Việt.", ["sinh nhật"]),
    ("Dịch sang tiếng Anh: 'Tôi thích đọc sách'.", ["read", "book"]),
    ("Sông nào dài nhất Việt Nam?", ["Đồng Nai", "Mê Kông", "Cửu Long"]),
    ("2 mũ 10 bằng bao nhiêu?", ["1024"]),
    ("Nêu một lợi ích của việc tập thể dục.", ["sức khỏe"]),
    ("Ai là tác giả của Truyện Kiều?", ["Nguyễn Du"]),
    ("Một năm có bao nhiêu tháng?", ["12"]),
    ("Nước sôi ở bao nhiêu độ C tại áp suất thường?", ["100"]),
    ("Kể tên một loại trái cây nhiệt đới.", ["xoài", "chuối", "dứa", "sầu riêng", "mít"]),
    ("Hãy tóm tắt ý nghĩa của câu 'Có công mài sắt có ngày nên kim'.", ["kiên trì", "cố gắng", "bền"]),
    ("Thành phố Hồ Chí Minh trước đây có tên là gì?", ["Sài Gòn"]),
    ("Giải thích ngắn gọn quang hợp là gì.", ["ánh sáng", "cây"]),
]


def main() -> None:
    rng = random.Random(SEED)
    DATA.mkdir(exist_ok=True)

    seen: set[str] = set()
    tickets: list[dict] = []
    while len(tickets) < 320:
        t = make_ticket(rng)
        if t["ticket"] in seen:
            continue
        seen.add(t["ticket"])
        tickets.append(t)

    train = [to_record(t) for t in tickets[:250]]
    target_eval = [to_record(t) for t in tickets[250:300]]
    holdout = [to_record(t) for t in tickets[300:]]

    regression = [
        {"instruction": q, "input": "", "keywords": kws}
        for q, kws in REGRESSION_ITEMS
    ]

    _write(DATA / "train_seed.jsonl", train)
    _write(DATA / "eval_target.jsonl", target_eval)
    _write(DATA / "eval_regression.jsonl", regression)
    _write(DATA / "holdout_secret.jsonl", holdout)

    # Decontamination check: no eval ticket may appear in the training set.
    train_bodies = {r["input"] for r in train}
    overlap = [r for r in target_eval + holdout if r["input"] in train_bodies]
    assert not overlap, f"CONTAMINATION: {len(overlap)} eval items leaked into train"

    print(f"train_seed        {len(train):>4}")
    print(f"eval_target       {len(target_eval):>4}")
    print(f"eval_regression   {len(regression):>4}")
    print(f"holdout_secret    {len(holdout):>4}")
    print("decontamination   OK (no eval ticket appears in train)")


def _write(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
