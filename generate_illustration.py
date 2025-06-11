from PIL import Image, ImageDraw, ImageFont

def generate_schematic_output(output_path="sample_output_illustration.png"):
    """
    Generates a schematic illustration of the project's output.
    """
    # Image settings
    img_width = 800
    img_height = 600
    bg_color = (255, 255, 255)  # White
    text_color = (0, 0, 0)      # Black
    car_body_color = (80, 80, 80)    # Dark Gray
    plate_bg_color = (200, 200, 200) # Light Gray
    car_bbox_color = (0, 128, 0)     # Green
    plate_bbox_color = (0, 0, 255)   # Blue

    # Create a new image
    image = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(image)

    # Attempt to load a font
    font_size_large = 24
    font_size_medium = 18
    font_size_small = 16
    try:
        # Try a common font, adjust path if necessary or use a generic name
        font_large = ImageFont.truetype("arial.ttf", font_size_large)
        font_medium = ImageFont.truetype("arial.ttf", font_size_medium)
        font_small = ImageFont.truetype("arial.ttf", font_size_small)
    except IOError:
        print("Arial font not found. Using Pillow's default bitmap font.")
        # Pillow's default font is a bitmap font, so size control is limited
        # For bitmap fonts, size is not passed to truetype, but it has a fixed size.
        # We can still try to get a default font instance.
        try:
            font_large = ImageFont.load_default() # No size argument for default bitmap font
            font_medium = font_large
            font_small = font_large
            print("Note: Default bitmap font does not support custom sizes well. Text scaling might be off.")
        except Exception as e:
            print(f"Could not load default font: {e}. Text might not be rendered.")
            font_large = None # No font available
            font_medium = None
            font_small = None


    # 1. Draw a "Car"
    # car_x1, car_y1, car_x2, car_y2
    car_x1 = img_width // 4
    car_y1 = img_height // 2 - 50
    car_x2 = img_width * 3 // 4
    car_y2 = img_height // 2 + 100
    draw.rectangle([car_x1, car_y1, car_x2, car_y2], fill=car_body_color)

    # 2. Draw a "License Plate" on the car
    plate_width = (car_x2 - car_x1) // 3
    plate_height = 40
    plate_x1 = car_x1 + (car_x2 - car_x1 - plate_width) // 2 # Centered on car body
    plate_y1 = car_y2 - plate_height - 20 # Near bottom of car body
    plate_x2 = plate_x1 + plate_width
    plate_y2 = plate_y1 + plate_height
    draw.rectangle([plate_x1, plate_y1, plate_x2, plate_y2], fill=plate_bg_color)

    # 3. Draw Bounding Boxes
    # Car BBox (slightly outside the car body for visibility)
    car_bbox_padding = 5
    draw.rectangle(
        [car_x1 - car_bbox_padding, car_y1 - car_bbox_padding,
         car_x2 + car_bbox_padding, car_y2 + car_bbox_padding],
        outline=car_bbox_color, width=3
    )
    # License Plate BBox (slightly outside the plate for visibility)
    plate_bbox_padding = 3
    draw.rectangle(
        [plate_x1 - plate_bbox_padding, plate_y1 - plate_bbox_padding,
         plate_x2 + plate_bbox_padding, plate_y2 + plate_bbox_padding],
        outline=plate_bbox_color, width=2
    )

    # 4. Add Text Annotations
    # Text for Vehicle ID and Speed (above car bbox)
    vehicle_text_y_pos = car_y1 - car_bbox_padding - font_size_large - 5 # 5px margin
    if font_large:
        draw.text((car_x1, vehicle_text_y_pos - font_size_medium - 2), "Vehicle ID: 1", fill=text_color, font=font_medium)
        draw.text((car_x1, vehicle_text_y_pos), "Speed: 25 px/frame", fill=text_color, font=font_medium)

    # Text for License Plate (below plate bbox)
    plate_text_y_pos = plate_y2 + plate_bbox_padding + 10 # 10px margin
    if font_medium:
        # Center plate text approximately
        plate_text = "PLATE: XYZ789"
        # text_width, text_height = draw.textsize(plate_text, font=font_medium) # Deprecated
        try:
            text_bbox = draw.textbbox((0,0), plate_text, font=font_medium)
            text_width = text_bbox[2] - text_bbox[0]
        except AttributeError: # Fallback for older Pillow if textbbox not present on draw
            text_width = font_medium.getsize(plate_text)[0] if hasattr(font_medium, 'getsize') else 80


        plate_text_x_pos = plate_x1 + (plate_width - text_width) // 2
        if plate_text_x_pos < 0 : plate_text_x_pos = plate_x1 # ensure not off screen
        draw.text((plate_text_x_pos, plate_text_y_pos), plate_text, fill=text_color, font=font_medium)

    # Add a title to the image
    if font_large:
        title_text = "Sample Output Annotation"
        try:
            title_bbox = draw.textbbox((0,0),title_text, font=font_large)
            title_width = title_bbox[2] - title_bbox[0]
        except AttributeError:
             title_width = font_large.getsize(title_text)[0] if hasattr(font_large, 'getsize') else 200

        title_x = (img_width - title_width) // 2
        draw.text((title_x, 20), title_text, fill=text_color, font=font_large)


    # 5. Save the image
    try:
        image.save(output_path)
        print(f"Illustration saved to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    generate_schematic_output()
