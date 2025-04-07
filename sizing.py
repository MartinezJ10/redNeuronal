from PIL import Image
import os

size_width = 600
size_height = 800

input_path = r"C:\Users\marti\Documents\Carrera jaja\12_DUODECIMO_TRIMESTRE_2025\IA\project\data"
output_path = os.path.join(input_path, "resized")

os.makedirs(output_path, exist_ok=True)

valid_extensions = ['.jpg', '.jpeg', '.png']

for item in os.listdir(input_path):
    full_input_path = os.path.join(input_path, item)
    img_name, ext = os.path.splitext(item)

    if ext.lower() not in valid_extensions:
        continue

    output_filename = f"resized_new_{img_name}_resized.jpg"
    full_output_path = os.path.join(output_path, output_filename)

    if os.path.exists(full_output_path):
        print(f'{item} is already resized')
        continue

    if os.path.isfile(full_input_path):
        print(f'Resizing {item}...')
        try:
            im = Image.open(full_input_path)
            if im.mode == 'RGBA':
                im = im.convert('RGB')
            imResize = im.resize((size_width, size_height), Image.Resampling.LANCZOS)
            imResize.save(full_output_path, 'JPEG', quality=95)
        except Exception as e:
            print(f"Failed to process {item}: {e}")

print('All done!')
