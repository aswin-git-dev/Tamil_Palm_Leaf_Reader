import base64

# Path to your image
img_path = 'images/vatta_button.png'
with open(img_path, 'rb') as f:
    img_bytes = f.read()
img_base64 = base64.b64encode(img_bytes).decode()
print(img_base64)
