import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List

app = FastAPI()

def process_image(image_bytes: bytes) -> List[str]:
    # 1. Read image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Binarize (Threshold)
    # Using inverse binary threshold assuming black frames on white background
    # Gray > 200 becomes 0 (Black background), Gray <= 200 becomes 255 (White frames)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 4. Find Contours with Hierarchy
    # Use RETR_CCOMP to get a 2-level hierarchy (External + Internal Holes)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    candidate_panels = []

    # Iterate through contours to find the frames
    # Hierarchy format: [Next, Previous, First_Child, Parent]
    for i, contour in enumerate(contours):
        # We are looking for the OUTER frame of a panel.
        # It should have a valid child (the inner edge of the frame) or be a solid block.
        # But mostly comic frames are drawn as lines, so they are double contours in CCOMP.
        
        # Filter by area to avoid noise
        if cv2.contourArea(contour) < 5000:  # Adjust threshold based on expected image size
            continue

        # Check if it has a child (First_Child != -1)
        child_idx = hierarchy[i][2]
        
        if child_idx != -1:
            # It has a child, meaning it's likely a frame.
            # We want to crop to the CHILD's bounding box (the inner content).
            inner_contour = contours[child_idx]
            candidate_panels.append(inner_contour)
        else:
            # It has no child. It might be a solid black block or the frame line is too thin/broken.
            # In this case, we use the contour itself but maybe shrink it slightly?
            # For now, let's treat it as a valid panel candidate check its parent.
            # If it has a parent, it might be the inner content itself!
            parent_idx = hierarchy[i][3]
            if parent_idx != -1:
                # It is an inner contour. We might have already added it via the parent check above?
                # The parent check ensures we only add unique panels. 
                # If we iterate all, we might double count.
                # Let's stick to: "Find Outer Frame -> Use Inner Box".
                pass
            else:
                 # Top level contour with no child.
                 # Could be a panel without a clear border (just content).
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

    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 6. Crop the image
        # Add a small padding? The requirements said "precisely crop the 4 regions based on contours".
        crop = img[y:y+h, x:x+w]

        # 7. Compress to WebP
        # Quality 80 is a good default for WebP
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), 80]
        success, encoded_img = cv2.imencode('.webp', crop, encode_param)
        
        if not success:
            continue
            
        # 8. Convert to Base64
        base64_str = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
        cropped_images_base64.append(base64_str)

    return cropped_images_base64

@app.post("/crop", response_model=List[str])
async def crop_panels(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    
    try:
        cropped_panels = process_image(content)
        # Returns a JSON array of strings
        return cropped_panels
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Comic Cropper API. Use POST /crop to upload an image."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
