import requests
import base64
import os
import sys

def test_crop(image_path):
    url = "http://127.0.0.1:8000/crop"
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    print("Uploading image...")
    with open(image_path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(url, files=files)
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the server. Make sure 'uvicorn main:app' is running.")
            return

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
                filename = os.path.join(output_dir, f"crop_{i+1}.webp")
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving image {i+1}: {e}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <path_to_image>")
    else:
        test_crop(sys.argv[1])
