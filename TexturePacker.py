import numpy as np
from PIL import Image
from tkinter import Tk, Label, Button, filedialog
import cv2

def pack_rectangles(image_paths, output_path, status_label):
    all_rects = []
    all_images = []

    total_images = len(image_paths)

    for image_path in image_paths:
        # Update status
        status_label.config(text=f"Processing {image_path}")

        # Load the image
        image = Image.open(image_path)
        image_np = np.array(image)

        # Add an alpha channel if it doesn't exist
        if image_np.shape[2] == 3:
            alpha_channel = np.ones((image_np.shape[0], image_np.shape[1], 1), dtype=np.uint8) * 255
            image_np = np.concatenate((image_np, alpha_channel), axis=2)
        
        all_images.append(image_np)

        # Check if the image is fully opaque
        if np.all(image_np[:, :, 3] == 255):
            # Treat the entire image as a single rectangle
            rects = [(0, 0, image_np.shape[1], image_np.shape[0])]  # Use the height and width of the image
        else:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get bounding rectangles for each contour
            rects = [cv2.boundingRect(cnt) for cnt in contours]

            # Ensure at least one rectangle
            if not rects:
                rects = [(0, 0, image_np.shape[1], image_np.shape[0])]

        # Store rectangles with their corresponding image index
        all_rects.extend([(rect, len(all_images) - 1) for rect in rects])

    # Sort rectangles by height (largest to smallest)
    all_rects = sorted(all_rects, key=lambda x: x[0][3], reverse=True)

    # Initialize free rectangles with one big rectangle
    max_width = max(image.shape[1] for image in all_images)
    max_height = sum(image.shape[0] for image in all_images)
    free_rects = [(0, 0, max_width, max_height)]

    packed_positions = []

    for rect, img_idx in all_rects:
        w, h = rect[2], rect[3]
        best_free_rect = None
        best_free_rect_idx = -1
        best_fit = None

        # Find the best fitting free rectangle
        for i, free_rect in enumerate(free_rects):
            fw, fh = free_rect[2], free_rect[3]
            if fw >= w and fh >= h:
                fit = (fw - w) * (fh - h)
                if best_fit is None or fit < best_fit:
                    best_fit = fit
                    best_free_rect = free_rect
                    best_free_rect_idx = i

        if best_free_rect is None:
            continue

        # Place the rectangle in the best fitting free rectangle
        fx, fy, fw, fh = best_free_rect
        packed_positions.append((fx, fy, w, h, rect[0], rect[1], img_idx))

        # Split the free rectangle
        new_free_rects = []
        new_free_rects.append((fx + w, fy, fw - w, h))  # Right part
        new_free_rects.append((fx, fy + h, fw, fh - h))  # Bottom part

        # Replace the used free rectangle
        free_rects.pop(best_free_rect_idx)
        for new_free_rect in new_free_rects:
            if new_free_rect[2] > 0 and new_free_rect[3] > 0:
                free_rects.append(new_free_rect)

    # Create a new packed image with the calculated size
    max_width = max(px + pw for px, py, pw, ph, ox, oy, img_idx in packed_positions)
    max_height = max(py + ph for px, py, pw, ph, ox, oy, img_idx in packed_positions)
    packed_image = np.zeros((max_height, max_width, 4), dtype=np.uint8)

    # Place rectangles into the packed image
    for px, py, pw, ph, ox, oy, img_idx in packed_positions:
        packed_image[py:py+ph, px:px+pw] = all_images[img_idx][oy:oy+ph, ox:ox+pw]

    # Convert back to Image
    packed_image_pil = Image.fromarray(packed_image)

    # Save the packed image
    packed_image_pil.save(output_path)
    print(f"Packed image saved to {output_path}")

    # Update status
    status_label.config(text=f"Packed image saved to {output_path}")

def open_file_dialog():
    file_paths = filedialog.askopenfilenames(filetypes=[
        ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.gif"),
        ("PNG files", "*.png"),
        ("JPEG files", "*.jpg;*.jpeg"),
        ("BMP files", "*.bmp"),
        ("TIFF files", "*.tiff"),
        ("GIF files", "*.gif"),
        ("All files", "*.*")
    ])
    if file_paths:
        output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("BMP files", "*.bmp"),
            ("TIFF files", "*.tiff"),
            ("GIF files", "*.gif"),
            ("All files", "*.*")
        ])
        if output_path:
            pack_rectangles(file_paths, output_path, status_label)

if __name__ == "__main__":
    root = Tk()
    root.title("Image Packer")
    root.geometry("400x200")

    label = Label(root, text="Select image files to pack", wraplength=350, justify="center")
    label.pack(pady=10)

    button = Button(root, text="Browse", command=open_file_dialog)
    button.pack(pady=10)

    status_label = Label(root, text="", wraplength=350, justify="center")
    status_label.pack(pady=10)

    root.mainloop()