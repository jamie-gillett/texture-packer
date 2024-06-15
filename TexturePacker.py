import numpy as np
from PIL import Image
from tkinter import Tk, Label, Button, filedialog
import cv2

class Packer:
    def __init__(self, verbose=True) -> None:
        self.verbose = verbose

    def load_images(self, image_paths):
        image_arrays = []
        for image_path in image_paths:
            image = Image.open(image_path)
            image_array = np.array(image)
            # Add an alpha channel if it doesn't exist
            if image_array.shape[2] == 3:
                alpha_channel = np.ones((image_array.shape[0], image_array.shape[1], 1), dtype=np.uint8) * 255
                image_array = np.concatenate((image_array, alpha_channel), axis=2)
            image_arrays.append(image_array)
        return image_arrays

    def identify_bounding_boxes(self, image_arrays):
        bounding_boxes = []
        for i, image_array in enumerate(image_arrays):
            # Check if the image is fully opaque
            if np.all(image_array[:, :, 3] == 255):
                # Treat the entire image as a single box
                boxes = [((0, 0, image_array.shape[1], image_array.shape[0]),i)]  # x-offset, y-offset, width, height
            else:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
                _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = [((x,y,w,h),i) for x,y,w,h in [cv2.boundingRect(contour) for contour in contours]]
                if not boxes:
                    boxes = [((0, 0, image_array.shape[1], image_array.shape[0]),i)] # Ensure at least one bounding box
            bounding_boxes.extend(boxes)
        bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0][3], reverse=True) # Sort bounding_boxes by height (largest to smallest)
        return bounding_boxes

    def pack(self, all_images, bounding_boxes):
        packed_positions = []
        # Initialize free rectangles with one big rectangle
        # TODO possible extract new image size function and pass in size as an argument
        max_width = max(image.shape[1] for image in all_images)
        max_height = sum(image.shape[0] for image in all_images) # If the height is based on the images and not the boxes, can I break this?
        empty_areas = [(0, 0, max_width, max_height)]
        for box, image_index in bounding_boxes:
            w, h = box[2], box[3]
            best_empty_area = None
            best_empty_area_index = -1
            best_fit = None
            # Find the best fitting free rectangle
            for i, empty_area in enumerate(empty_areas):
                fw, fh = empty_area[2], empty_area[3]
                if fw >= w and fh >= h:
                    fit = (fw - w) * (fh - h)
                    if best_fit is None or fit < best_fit:
                        best_fit = fit
                        best_empty_area = empty_area
                        best_empty_area_index = i
            if best_empty_area is None:
                # how would you get here?!
                continue
            # Place the rectangle in the best fitting free rectangle
            fx, fy, fw, fh = best_empty_area
            packed_positions.append((fx, fy, w, h, box[0], box[1], image_index))
            # Split the free rectangle
            new_free_rects = []
            new_free_rects.append((fx + w, fy, fw - w, h))  # Right part
            new_free_rects.append((fx, fy + h, fw, fh - h))  # Bottom part
            # Replace the used free rectangle
            empty_areas.pop(best_empty_area_index)
            for new_free_rect in new_free_rects:
                if new_free_rect[2] > 0 and new_free_rect[3] > 0:
                    empty_areas.append(new_free_rect)
        return packed_positions

    def generate_image(self, all_images, packed_positions):
        # Create a new packed image with the calculated size
        max_width = max(px + pw for px, py, pw, ph, ox, oy, img_idx in packed_positions)
        max_height = max(py + ph for px, py, pw, ph, ox, oy, img_idx in packed_positions)
        packed_image_np = np.zeros((max_height, max_width, 4), dtype=np.uint8)
        # Place rectangles into the packed image
        for px, py, pw, ph, ox, oy, img_idx in packed_positions:
            packed_image_np[py:py+ph, px:px+pw] = all_images[img_idx][oy:oy+ph, ox:ox+pw]
        packed_image = Image.fromarray(packed_image_np) # Convert back to Image
        return packed_image

    def process_textures(self, image_paths):
        all_images = self.load_images(image_paths)
        bounding_boxes = self.identify_bounding_boxes(all_images)
        packed_positions = self.pack(all_images, bounding_boxes)
        output_image = self.generate_image(all_images, packed_positions)
        return output_image



class PackerGUI:
    def __init__(self) -> None:
        self.filepaths = None
        self.output_path = None
        self.root = self.app_setup()
        self.packer = Packer()
        
    def app_setup(self):
        root = Tk()
        root.title("Texture Packer")
        root.geometry("400x250")

        self.input_label = Label(root, text="Select image files to pack", wraplength=350, justify="center")
        self.input_label.pack(pady=10)

        self.open_button = Button(root, text="Select Files", command=self.open_files)
        self.open_button.pack(pady=10)
        
        self.output_label = Label(root, text="Select output path", wraplength=350, justify="center")
        self.output_label.pack(pady=10)

        self.savepath_button = Button(root, text="Save Location", command=self.save_path)
        self.savepath_button.pack(pady=10)

        self.run_button = Button(root, text="Run", command=self.run)
        self.run_button.pack(pady=10)
        
        return root
    
    def start(self):
        self.root.mainloop()

    def open_files(self):
        self.filepaths = filedialog.askopenfilenames(filetypes=[
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.gif"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("BMP files", "*.bmp"),
            ("TIFF files", "*.tiff"),
            ("GIF files", "*.gif"),
            ("All files", "*.*")
        ])
        if self.filepaths:
            self.input_label.config(text="Image Files Selected")     
        
    def save_path(self):
        # TODO: Perhaps separate out the saving to be done after the processing (add check for self.output_image)
        self.output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("BMP files", "*.bmp"),
            ("TIFF files", "*.tiff"),
            ("GIF files", "*.gif"),
            ("All files", "*.*")
        ])
        if self.output_path:
            self.output_label.config(text="Output Path Selected")

    def run(self):
        if not (self.filepaths and self.output_path):
            print("<!> Need to select both input images and output path.")
            return
        self.output_image = self.packer.process_textures(self.filepaths) # saved as instance variable in case decide to separate saving
        self.output_image.save(self.output_path)
        print("Job done.")
        self.filepaths = None
        self.input_label.config(text="Select Files")
        self.output_path = None
        self.output_label.config(text="Select output path")


if __name__ == "__main__":
    app = PackerGUI()
    app.start()