# Image Cut API

這是一個基於 FastAPI 與 OpenCV 的四格漫畫裁切服務。它可以自動偵測並裁切四格漫畫中的 4 個主要面板，去除黑框，並回傳 WebP 格式的圖片（Base64 編碼）。

## 功能特色
- **自動偵測**：使用 OpenCV `RETR_CCOMP` 尋找漫畫格子的內框，自動去除黑邊。
- **智慧排序**：自動將裁切後的圖片依照「由上而下、由左而右」的順序排列。
- **高效傳輸**：裁切後的圖片壓縮為 `WebP` 格式並轉為 Base64 字串，方便前端直接使用。
- **雙重模式**：支援「上傳圖片檔案」與「提供圖片網址」兩種方式。

## 安裝與執行

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 啟動伺服器
```bash
uvicorn main:app --reload
# 或指定 port
uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## API 文件 (Base URL: `https://image-cut.zeabur.app`)

### 1. 上傳圖片裁切
將本地圖片檔案上傳進行處理。

- **Endpoint**: `POST /crop`
- **Content-Type**: `multipart/form-data`

**cURL 範例**:
```bash
curl -X POST https://image-cut.zeabur.app/crop \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/comic.jpg"
```

**參數**:
- `file`: 圖片檔案 (Binary)

**回傳範例 (JSON)**:
```json
[
  "UklGRuZ...", // Base64 string of image 1
  "UklGRuZ...", // Base64 string of image 2
  "UklGRuZ...", // Base64 string of image 3
  "UklGRuZ..."  // Base64 string of image 4
]
```

---

### 2. 網址圖片裁切
提供圖片的 URL 進行處理。

- **Endpoint**: `POST /crop/url`
- **Content-Type**: `application/json`

**cURL 範例**:
```bash
curl -X POST https://image-cut.zeabur.app/crop/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/comic.jpg"
  }'
```

**Body**:
```json
{
  "url": "https://example.com/comic.jpg"
}
```

**回傳範例 (JSON)**:
與 `/crop` 相同，回傳 Base64 字串陣列。

---

## 測試腳本
專案內附帶 `test_client.py` 可供測試。

**測試檔案上傳**:
```bash
python3 test_client.py file path/to/comic.jpg
```

**測試網址**:
```bash
python3 test_client.py url https://example.com/comic.jpg
```

裁切後的結果會儲存在 `output_crops/` 資料夾中。
