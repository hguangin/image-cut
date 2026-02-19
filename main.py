import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List

app = FastAPI()

def process_image(image_bytes: bytes, out_format: str = "webp", quality: int = 80) -> List[str]:
    # Validate format
    valid_formats = {"webp", "jpeg", "png", "jpg"}
    out_format = out_format.lower()
    if out_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {out_format}. Use webp, jpeg, or png.")
    
    if out_format == "jpg":
        out_format = "jpeg"

    # Clamp quality to 1-100
    quality = max(1, min(100, quality))

    # 1. Read image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_h, img_w = gray.shape
    img_area = img_h * img_w

    # 3. 邊緣偵測 (Canny Edge Detection)
    # 先做一點模糊化，減少圖片內部的雜訊與紋理干擾
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 使用 Canny 找出所有物體、框線的邊緣
    edges = cv2.Canny(blurred, 30, 150)

    # 4. 膨脹邊緣 (Dilate)
    # 有些黑框可能不連續或有缺口，透過膨脹可以把它們連起來
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # 5. 尋找所有輪廓 (不分階層)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 6. 利用邊界框 (Bounding Box) 面積初步過濾
    # 四格漫畫的其中一格，面積通常介於整張圖的 3% 到 85% 之間
    candidate_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # 排除太小的雜點（如文字、人臉）與太大的整張圖外框
        if img_area * 0.03 < area < img_area * 0.85:
            candidate_boxes.append((x, y, w, h, area))

    # 依面積由大到小排序
    candidate_boxes.sort(key=lambda b: b[4], reverse=True)

    # 7. 非極大值抑制 (NMS, Non-Maximum Suppression)
    # 漫畫最外層的框一定最先被挑選。而框內部的次要物件（如對話框）雖然有自己的框，
    # 但會完全重疊在大框裡面，這個步驟會把它們徹底剔除。
    final_boxes = []
    for box in candidate_boxes:
        x, y, w, h, area = box
        
        is_overlap = False
        for fbox in final_boxes:
            fx, fy, fw, fh = fbox
            
            # 計算交集範圍
            ix_min = max(x, fx)
            iy_min = max(y, fy)
            ix_max = min(x + w, fx + fw)
            iy_max = min(y + h, fy + fh)
            
            # 如果有交集
            if ix_max > ix_min and iy_max > iy_min:
                inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                # 如果重疊面積大於「當前處理框（較小者）」的 30%
                # 代表這個框是已經存在的大框的附屬物，必須剔除
                if inter_area > 0.3 * area:
                    is_overlap = True
                    break
                    
        if not is_overlap:
            final_boxes.append((x, y, w, h))
            
    # 取最大的四塊 (確保就算遇到五格也只取主要的四格)
    final_boxes = final_boxes[:4]

    # 8. 空間排序 (由上而下、由左至右)
    def sort_key(box):
        x, y, w, h = box
        # 將 Y 座標取概數（約 10% 圖片高度為一個區間），讓同一排的格子能被分在同一組
        row_tolerance = max(10, int(img_h * 0.1))
        return ((y // row_tolerance) * row_tolerance, x)
    
    final_boxes.sort(key=sort_key)

    cropped_images_base64 = []
import io
from PIL import Image

def process_image(image_bytes: bytes, out_format: str = "webp", quality: int = 80) -> List[str]:
    # Validate format
    valid_formats = {"webp", "jpeg", "png", "jpg"}
    out_format = out_format.lower()
    # ... ignoring validation and canny setup in this patch ...
    
    # 8. 空間排序 (由上而下、由左至右)
    def sort_key(box):
        x, y, w, h = box
        # 將 Y 座標取概數（約 10% 圖片高度為一個區間），讓同一排的格子能被分在同一組
        row_tolerance = max(10, int(img_h * 0.1))
        return ((y // row_tolerance) * row_tolerance, x)
    
    final_boxes.sort(key=sort_key)

    cropped_images_base64 = []

    for (x, y, w, h) in final_boxes:
        # 動態內縮 (Padding)：為了徹底切掉可能包含在外的黑框線或白邊
        # 使用動態比例內縮 (寬度或高度的 1.5%，但最少 5 像素)
        pad_x = max(5, int(w * 0.015))
        pad_y = max(5, int(h * 0.015))
        
        # 確保裁切範圍不會變成負數或出界
        crop_y_start = min(y + pad_y, y + h)
        crop_y_end = max(y + h - pad_y, crop_y_start)
        crop_x_start = min(x + pad_x, x + w)
        crop_x_end = max(x + w - pad_x, crop_x_start)
        
        # 進行裁切
        crop = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # 將 OpenCV 的 BGR 轉換為 Pillow 支援的 RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        # 準備記憶體快取來接產生二進位資料
        buffer = io.BytesIO()

        # 根據請求格式套入 Pillow 最極限的壓縮引數
        if out_format == "webp":
            # method=6 代表讓編碼器花最多的時間去找尋「最小檔案」的路徑
            pil_img.save(buffer, format="WEBP", quality=quality, method=6)
        elif out_format == "jpeg":
            # optimize=True: 優化霍夫曼編碼表
            # progressive=True: 漸進式網路載入
            # subsampling=0: 4:4:4 無損色彩取樣 (防止漫畫線條色溢)
            pil_img.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=0)
        elif out_format == "png":
            # PNG quality map
            compression_level = max(0, min(9, round(9 * (100 - quality) / 100)))
            # optimize=True: 疊加 zlib 多種嘗試找最小解
            pil_img.save(buffer, format="PNG", optimize=True, compress_level=compression_level)

        # 轉 Base64 回傳
        encoded_bytes = buffer.getvalue()
        base64_str = base64.b64encode(encoded_bytes).decode('utf-8')
        cropped_images_base64.append(base64_str)

    return cropped_images_base64


from pydantic import BaseModel
from fastapi import Form
import requests

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
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    
    try:
        cropped_panels = process_image(content, format, quality)
        # Returns a JSON array of strings
        return cropped_panels
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/crop/url", response_model=List[str])
async def crop_panels_from_url(image: ImageURL):
    try:
        response = requests.get(image.url, timeout=10)
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
             raise HTTPException(status_code=400, detail="URL does not point to a valid image")

        content = response.content
        cropped_panels = process_image(content, image.format, image.quality)
        return cropped_panels
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Comic Cropper API. Use POST /crop to upload an image or POST /crop/url to provide an image URL."}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
