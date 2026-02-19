import requests
import base64
import os
import sys
import argparse

def test_crop_url(image_url, out_format="webp", quality=80):
    url = "http://127.0.0.1:8000/crop/url"
    print(f"Requesting crop for URL: {image_url} (Format: {out_format}, Quality: {quality})")
    
    payload = {
        "url": image_url,
        "format": out_format,
        "quality": quality
    }
    
    try:
        response = requests.post(url, json=payload)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        return

    process_response(response, out_format)

def process_response(response, out_format):
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list):
            images = data
        else:
            images = data.get("images", [])
            
        print(f"Successfully processed! Received {len(images)} images.")
        
        output_dir = "output_crops"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for i, img_base64 in enumerate(images):
            try:
                img_data = base64.b64decode(img_base64)
                # Map jpg to jpeg for file extension if returned that way
                ext = out_format if out_format != "jpeg" else "jpg"
                filename = os.path.join(output_dir, f"crop_{i+1}.{ext}")
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving image {i+1}: {e}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_crop_file(image_path, out_format="webp", quality=80):
    url = "http://127.0.0.1:8000/crop"
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    print(f"Uploading image file: {image_path} (Format: {out_format}, Quality: {quality})")
    
    data = {"format": out_format, "quality": quality}
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(url, files=files, data=data)
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the server.")
            return
    process_response(response, out_format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test API for comic cropping")
    parser.add_argument("mode", choices=["file", "url"], help="Mode of testing: file upload or url")
    parser.add_argument("target", help="File path or URL")
    parser.add_argument("-f", "--format", default="webp", choices=["webp", "jpeg", "jpg", "png"], help="Output image format")
    parser.add_argument("-q", "--quality", type=int, default=80, help="Output image quality (1-100)")
    
    args = parser.parse_args()
    
    if args.mode == "file":
        test_crop_file(args.target, args.format, args.quality)
    elif args.mode == "url":
        test_crop_url(args.target, args.format, args.quality)
