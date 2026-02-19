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

    # 3. Binarize (Threshold)
    # 將閾值降到 80，這樣才能只捕捉到「真正的純黑邊框」，
    # 避免像暗色房間那種偏暗的背景也被判定為邊框而糊在一起。
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # 4. Find Contours with Hierarchy
    # 使用 RETR_CCOMP 來取得內外兩層的輪廓關係
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    candidate_panels = []

    # 5. 找出所有的內部視窗 (洞)
    for i, contour in enumerate(contours):
        # 過濾掉太小的雜點
        if cv2.contourArea(contour) < 5000:
            continue

        # 取得這個輪廓的「第一個子輪廓 (First Child)」的索引
        child_idx = hierarchy[i][2]
        
        if child_idx != -1:
            # 如果有子輪廓（代表這是一個有洞的框），必須迴圈找出「所有」子輪廓。
            # 因為四格漫畫的十字黑框是一個單一的連通區塊，裡面會同時包含 4 個洞（4 個畫面）。
            current_child = child_idx
            while current_child != -1:
                inner_contour = contours[current_child]
                # 過濾掉洞裡面太小的區塊
                if cv2.contourArea(inner_contour) > 5000:
                    candidate_panels.append(inner_contour)
                # 換到下一個兄弟節點（下一個洞）
                current_child = hierarchy[current_child][0]
        else:
            # 如果沒有子輪廓但本身夠大，且沒有父節點，當作備用候選（例如實心或者非框線的圖）
            parent_idx = hierarchy[i][3]
            if parent_idx == -1:
                 candidate_panels.append(contour)

    # Sort candidates by area to find the main 4
    candidate_panels.sort(key=cv2.contourArea, reverse=True)
    main_contours = candidate_panels[:4]

    # Sort the 4 contours spatially
    bounding_boxes = [cv2.boundingRect(c) for c in main_contours]
    
    # Custom sort: Round Y to nearest 50 pixels to group into rows, then sort by X
    def sort_key(box):
        x, y, w, h = box
        return (round(y / 50) * 50, x)
    
    # Zip contours with their boxes
    contours_with_boxes = list(zip(main_contours, bounding_boxes))
    # Sort based on the box
    contours_with_boxes.sort(key=lambda cb: sort_key(cb[1]))
    
    # Extract just the sorted contours
    sorted_contours = [cb[0] for cb in contours_with_boxes]

    cropped_images_base64 = []

    # Setup encoding parameters based on requested format
    encode_ext = f".{out_format}"
    encode_param = []
    if out_format == "webp":
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    elif out_format == "jpeg":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif out_format == "png":
        # PNG compression ranges from 0 (no compression, fast) to 9 (max compression, slow). 
        # It is lossless, so "quality" here just means file size vs processing time.
        # We map quality 1-100 to compression 9-0 (where quality 100 means compression 0, max filesize, fastest).
        compression = round(9 * (100 - quality) / 100)
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), compression]

    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 6. Crop the image
        # Add a small padding? The requirements said "precisely crop the 4 regions based on contours".
        crop = img[y:y+h, x:x+w]

        # 7. Compress
        success, encoded_img = cv2.imencode(encode_ext, crop, encode_param)
        
        if not success:
            continue
            
        # 8. Convert to Base64
        base64_str = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
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
