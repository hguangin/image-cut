import cv2
import numpy as np
import base64
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import requests
import io
from PIL import Image

app = FastAPI()

def process_image(image_bytes: bytes, out_format: str = "webp", quality: int = 80) -> List[str]:
    # 驗證格式
    valid_formats = {"webp", "jpeg", "png", "jpg"}
    out_format = out_format.lower()
    if out_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"不支援的格式: {out_format}")
    
    if out_format == "jpg":
        out_format = "jpeg"

    # 品質限制在 1-100
    quality = max(1, min(100, quality))

    # 1. 讀取影像
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="無效的影像檔案")

    # 2. 轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    img_area = img_h * img_w

    # 3. 邊緣偵測 (Canny)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)

    # 4. 形態學優化：改用「閉運算 (Closing)」連接斷線，避免線條過度膨脹導致裁到黑邊
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. 尋找輪廓：改用 RETR_CCOMP 獲取層級資訊，這有助於定位黑框的「內緣」
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    if hierarchy is not None:
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # 篩選條件：面積介於整圖的 3% ~ 85%
            if img_area * 0.03 < area < img_area * 0.85:
                # 如果該輪廓有父輪廓 (hierarchy[0][i][3] != -1)，表示它更可能是黑框內部的區域
                score = area
                if hierarchy[0][i][3] != -1:
                    score += img_area * 0.1  # 給予內層輪廓更高的優先級權重
                candidate_boxes.append((x, y, w, h, score))

    # 依分數由大到小排序
    candidate_boxes.sort(key=lambda b: b[4], reverse=True)

    # 6. 非極大值抑制 (NMS)
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
                # 若重疊面積大於當前框的 30%，視為重複
                if inter_area > 0.3 * (w * h):
                    is_overlap = True
                    break
        if not is_overlap:
            final_boxes.append((x, y, w, h))
            
    # 取主要的四格
    final_boxes = final_boxes[:4]

    # 7. 空間排序 (由上而下、由左至右)
    row_tolerance = max(10, int(img_h * 0.1))
    final_boxes.sort(key=lambda b: ((b[1] // row_tolerance) * row_tolerance, b[0]))

    cropped_images_base64 = []

    for (x, y, w, h) in final_boxes:
        # 8. 動態內縮 (Padding)：增加至 2% 寬高，為 JPEG 邊緣雜訊留緩衝
        pad_x = max(6, int(w * 0.02))
        pad_y = max(6, int(h * 0.02))
        
        y_s, y_e = max(0, y + pad_y), min(img_h, y + h - pad_y)
        x_s, x_e = max(0, x + pad_x), min(img_w, x + w - pad_x)
        
        crop = img[y_s:y_e, x_s:x_e]

        if crop.size == 0:
            continue

        # 9. 智慧邊緣清洗 (Smart Trim)：檢查四周是否有殘留黑邊並自動修剪
        # 轉換為灰階進行亮度分析
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # 臨界亮度 65 (接近黑色則繼續切)，最多向內再切 10 像素避免切掉內容
        trim_limit = 10
        for _ in range(trim_limit):
            if crop_gray.shape[1] > 20 and np.mean(crop_gray[:, 0]) < 65: # 左
                crop = crop[:, 1:]
                crop_gray = crop_gray[:, 1:]
            else: break
            
        for _ in range(trim_limit):
            if crop_gray.shape[1] > 20 and np.mean(crop_gray[:, -1]) < 65: # 右
                crop = crop[:, :-1]
                crop_gray = crop_gray[:, :-1]
            else: break

        # 將 OpenCV (BGR) 轉為 Pillow (RGB)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        buffer = io.BytesIO()

        # 針對輸出格式優化
        if out_format == "webp":
            pil_img.save(buffer, format="WEBP", quality=quality, method=6)
        elif out_format == "jpeg":
            # 針對 JPEG 漫畫：優化編碼、不使用色彩降採樣 (4:4:4)
            pil_img.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=0)
        elif out_format == "png":
            compression_level = max(0, min(9, round(9 * (100 - quality) / 100)))
            pil_img.save(buffer, format="PNG", optimize=True, compress_level=compression_level)

        encoded_bytes = buffer.getvalue()
        base64_str = base64.b64encode(encoded_bytes).decode('utf-8')
        cropped_images_base64.append(base64_str)

    return cropped_images_base64

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
