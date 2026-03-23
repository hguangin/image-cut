import cv2
import numpy as np
import base64
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import requests
import io
from PIL import Image

app = FastAPI()


# ============================================================
# 方法 A：分隔線偵測法（優先使用）
# 適用：格子間有明顯暗色分隔線的四宮格漫畫
# ============================================================
def detect_by_separator_lines(gray: np.ndarray, img: np.ndarray) -> Optional[List[np.ndarray]]:
    """
    掃描整張圖的水平/垂直亮度分佈，找出貫穿全圖的暗線（分隔線），
    再依分隔線位置精準切割四宮格。
    
    支援兩種漫畫結構：
    - 有外框 + 中間雙暗線分隔（正常黑框漫畫）
    - 無外框 / 薄外框 + 中間單暗線分隔（邊緣到邊緣的漫畫）
    
    回傳 4 張裁切圖（左上、右上、左下、右下），或 None 表示偵測失敗。
    """
    h, w = gray.shape

    # 計算每一行/列的平均亮度
    col_profile = np.mean(gray, axis=0)  # shape: (w,)  每列x的平均亮度
    row_profile = np.mean(gray, axis=1)  # shape: (h,)  每行y的平均亮度

    def find_dark_segments(profile: np.ndarray, threshold: int = 50) -> List[Tuple[int, int]]:
        """
        找出亮度低於閾值的連續暗區段。
        threshold 設為 50（而非 80），避免漫畫中的暗色場景
        （如夜景、暗房）被誤判為分隔線。真正的分隔線整列/整行
        平均亮度通常在 10~40 之間，遠低於 50。
        """
        segments = []
        in_dark = False
        start = 0
        for i in range(len(profile)):
            if profile[i] < threshold:
                if not in_dark:
                    start = i
                    in_dark = True
            else:
                if in_dark:
                    segments.append((start, i - 1))
                    in_dark = False
        if in_dark:
            segments.append((start, len(profile) - 1))
        return segments

    def pick_separator_lines(
        inner_segs: List[Tuple[int, int]], dim_size: int
    ) -> List[Tuple[int, int]]:
        """
        從中間暗帶中挑出最可能是分隔線的 1~2 條。
        策略：優先選最靠近圖片正中間的暗帶。
        如果有兩條暗帶彼此很近（< 圖寬/高 10%），視為同一組雙線分隔。
        """
        if len(inner_segs) <= 2:
            return inner_segs

        center = dim_size / 2
        scored = sorted(inner_segs, key=lambda seg: abs((seg[0] + seg[1]) / 2 - center))

        result = [scored[0]]
        if len(scored) > 1:
            gap = abs(scored[1][0] - scored[0][1])
            if gap < dim_size * 0.1:
                result.append(scored[1])
                result.sort()

        return result

    v_segs = find_dark_segments(col_profile)
    h_segs = find_dark_segments(row_profile)

    # 分類：位於圖片邊緣 10% 以內的視為「外框」，其餘為「中間分隔線」
    edge_ratio = 0.1
    v_outer = [(s, e) for s, e in v_segs if s < w * edge_ratio or e > w * (1 - edge_ratio)]
    v_inner = [(s, e) for s, e in v_segs if w * edge_ratio <= s and e <= w * (1 - edge_ratio)]
    h_outer = [(s, e) for s, e in h_segs if s < h * edge_ratio or e > h * (1 - edge_ratio)]
    h_inner = [(s, e) for s, e in h_segs if h * edge_ratio <= s and e <= h * (1 - edge_ratio)]

    # 至少要有 1 條垂直 + 1 條水平的中間分隔線才能切四宮格
    if len(v_inner) < 1 or len(h_inner) < 1:
        return None

    # 如果偵測到過多暗帶（可能是暗色場景干擾），挑選最靠近中心的
    v_inner = pick_separator_lines(v_inner, w)
    h_inner = pick_separator_lines(h_inner, h)

    # --- 決定四邊邊界 ---
    # 有外框 → 內容從外框線內側開始
    # 無外框 → 內容從圖片邊緣開始
    if v_outer and v_outer[0][0] < w * edge_ratio:
        left = v_outer[0][1] + 1
    else:
        left = 0

    if v_outer and v_outer[-1][1] > w * (1 - edge_ratio):
        right = v_outer[-1][0] - 1
    else:
        right = w - 1

    if h_outer and h_outer[0][0] < h * edge_ratio:
        top = h_outer[0][1] + 1
    else:
        top = 0

    if h_outer and h_outer[-1][1] > h * (1 - edge_ratio):
        bottom = h_outer[-1][0] - 1
    else:
        bottom = h - 1

    # --- 決定垂直分隔位置 ---
    if len(v_inner) >= 2:
        # 有黑框的正常情況：兩條線夾著間隙（左格右框線 + 右格左框線）
        v_split_left = v_inner[0][0] - 1      # 左半格的右邊界
        v_split_right = v_inner[-1][1] + 1     # 右半格的左邊界
    else:
        # 無黑框 / 薄分隔線：只有一條線
        v_split_left = v_inner[0][0] - 1
        v_split_right = v_inner[0][1] + 1

    # --- 決定水平分隔位置 ---
    if len(h_inner) >= 2:
        h_split_top = h_inner[0][0] - 1        # 上半格的下邊界
        h_split_bottom = h_inner[-1][1] + 1     # 下半格的上邊界
    else:
        h_split_top = h_inner[0][0] - 1
        h_split_bottom = h_inner[0][1] + 1

    # --- 裁切四格 ---
    panels = [
        img[top:h_split_top + 1, left:v_split_left + 1],          # 左上
        img[top:h_split_top + 1, v_split_right:right + 1],        # 右上
        img[h_split_bottom:bottom + 1, left:v_split_left + 1],    # 左下
        img[h_split_bottom:bottom + 1, v_split_right:right + 1],  # 右下
    ]

    # 驗證：每格面積至少佔整圖的 5%，且不為空
    img_area = h * w
    valid_panels = []
    for p in panels:
        if p.size == 0:
            return None
        if p.shape[0] * p.shape[1] < img_area * 0.05:
            return None
        valid_panels.append(p)

    return valid_panels


# ============================================================
# 方法 B：輪廓偵測法（Fallback）
# 適用：格子間沒有明顯直線分隔的漫畫，或非標準排版
# ============================================================
def detect_by_contours(gray: np.ndarray, img: np.ndarray) -> List[np.ndarray]:
    """
    使用 OpenCV 的 Canny + RETR_CCOMP 輪廓偵測，找出漫畫面板。
    這是原始演算法，作為分隔線偵測失敗時的後備方案。
    """
    img_h, img_w = gray.shape
    img_area = img_h * img_w

    # 邊緣偵測
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)

    # 形態學閉運算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 尋找輪廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    if hierarchy is not None:
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if img_area * 0.03 < area < img_area * 0.85:
                score = area
                if hierarchy[0][i][3] != -1:
                    score += img_area * 0.1
                candidate_boxes.append((x, y, w, h, score))

    candidate_boxes.sort(key=lambda b: b[4], reverse=True)

    # NMS
    final_boxes = []
    for box in candidate_boxes:
        x, y, w, h, _ = box
        is_overlap = False
        for fbox in final_boxes:
            fx, fy, fw, fh = fbox
            ix_min, iy_min = max(x, fx), max(y, fy)
            ix_max, iy_max = min(x + w, fx + fw), min(y + h, fy + fh)
            if ix_max > ix_min and iy_max > iy_min:
                inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                if inter_area > 0.3 * (w * h):
                    is_overlap = True
                    break
        if not is_overlap:
            final_boxes.append((x, y, w, h))

    final_boxes = final_boxes[:4]

    # 空間排序
    row_tolerance = max(10, int(img_h * 0.1))
    final_boxes.sort(key=lambda b: ((b[1] // row_tolerance) * row_tolerance, b[0]))

    panels = []
    for (x, y, w, h) in final_boxes:
        # 動態內縮
        pad_x = max(6, int(w * 0.02))
        pad_y = max(6, int(h * 0.02))
        y_s, y_e = max(0, y + pad_y), min(img_h, y + h - pad_y)
        x_s, x_e = max(0, x + pad_x), min(img_w, x + w - pad_x)
        crop = img[y_s:y_e, x_s:x_e]

        if crop.size == 0:
            continue

        # 邊緣清洗
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        trim_limit = 10
        for _ in range(trim_limit):
            if crop_gray.shape[1] > 20 and np.mean(crop_gray[:, 0]) < 65:
                crop = crop[:, 1:]
                crop_gray = crop_gray[:, 1:]
            else:
                break
        for _ in range(trim_limit):
            if crop_gray.shape[1] > 20 and np.mean(crop_gray[:, -1]) < 65:
                crop = crop[:, :-1]
                crop_gray = crop_gray[:, :-1]
            else:
                break

        panels.append(crop)

    return panels


# ============================================================
# 主處理函式
# ============================================================
def process_image(image_bytes: bytes, out_format: str = "webp", quality: int = 80) -> List[str]:
    # 驗證格式
    valid_formats = {"webp", "jpeg", "png", "jpg"}
    out_format = out_format.lower()
    if out_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"不支援的格式: {out_format}")

    if out_format == "jpg":
        out_format = "jpeg"

    quality = max(1, min(100, quality))

    # 讀取影像
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="無效的影像檔案")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 優先嘗試分隔線偵測法
    panels = detect_by_separator_lines(gray, img)

    # 若失敗，使用輪廓偵測法
    if panels is None or len(panels) != 4:
        panels = detect_by_contours(gray, img)

    # 編碼輸出
    cropped_images_base64 = []
    for crop in panels:
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        buffer = io.BytesIO()

        if out_format == "webp":
            pil_img.save(buffer, format="WEBP", quality=quality, method=6)
        elif out_format == "jpeg":
            pil_img.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=0)
        elif out_format == "png":
            compression_level = max(0, min(9, round(9 * (100 - quality) / 100)))
            pil_img.save(buffer, format="PNG", optimize=True, compress_level=compression_level)

        encoded_bytes = buffer.getvalue()
        base64_str = base64.b64encode(encoded_bytes).decode('utf-8')
        cropped_images_base64.append(base64_str)

    return cropped_images_base64


# ============================================================
# API 路由
# ============================================================
class ImageURL(BaseModel):
    url: str
    format: str = "webp"
    quality: int = 80


@app.post("/crop", response_model=List[str])
async def crop_panels(
    file: UploadFile = File(...),
    format: str = Form("webp"),
    quality: int = Form(80)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="必須上傳影像檔案")
    content = await file.read()
    try:
        return process_image(content, format, quality)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理失敗: {str(e)}")


@app.post("/crop/url", response_model=List[str])
async def crop_panels_from_url(image: ImageURL):
    try:
        response = requests.get(image.url, timeout=10)
        response.raise_for_status()
        return process_image(response.content, image.format, image.quality)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"網址處理失敗: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Comic Cropper API 運作中。"}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
