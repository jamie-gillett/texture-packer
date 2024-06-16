import numpy as np
from PIL import Image
import cv2

# TODO: Go through and clean up code
# TODO: Replace packing bounds calculation
# TODO: Replace packing algorithm
# TODO: Create proper print outs for verbose (incl. formatting)

class Packer:
    def __init__(self, verbose=True) -> None:
        self.verbose = verbose

    def load_images(self, image_paths):
        if self.verbose: print("load_images")
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
        if self.verbose: print("identify_bounding_boxes")
        bounding_boxes = []
        for i, image_array in enumerate(image_arrays):
            # Check if the image is fully opaque
            if np.all(image_array[:, :, 3] == 255):
                # Treat the entire image as a single box
                boxes = [((0, 0), (image_array.shape[1], image_array.shape[0]),i)]  # ((x-offset, y-offset), (width, height), image index)
            else:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
                _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = [((x,y),(w,h),i) for x,y,w,h in [cv2.boundingRect(contour) for contour in contours]]
                if not boxes:
                    boxes = [((0, 0), (image_array.shape[1], image_array.shape[0]),i)] # Ensure at least one bounding box
            bounding_boxes.extend(boxes)
        bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1][1], reverse=True) # Sort bounding_boxes by height (largest to smallest)
        coordinates, dimensions, image_indices = [_ for _ in list(zip(*bounding_boxes))]
        return coordinates, dimensions, image_indices
    
    def calculate_packing_bounds(self, images):
        if self.verbose: print("calculate_packing_bounds")
        width = max(image.shape[1] for image in images)
        height = sum(image.shape[0] for image in images)
        return width, height

    def pack(self, bounds, box_dimensions):
        if self.verbose: print("pack")
        new_coordinates = []
        max_width, max_height = bounds
        empty_areas = [((0, 0), (max_width, max_height))]
        for box in box_dimensions:
            box_width, box_height = box[0], box[1]
            best_empty_area = None
            best_empty_area_index = -1
            best_fit = None
            # Find the best fitting free rectangle
            for i, empty_area in enumerate(empty_areas):
                test_width, test_height = empty_area[1][0], empty_area[1][1]
                if test_width >= box_width and test_height >= box_height:
                    fit = (test_width - box_width) * (test_height - box_height)
                    if best_fit is None or fit < best_fit:
                        best_fit = fit
                        best_empty_area = empty_area
                        best_empty_area_index = i
            if best_empty_area is None:
                continue
            # Place the rectangle in the best fitting free rectangle
            coordinates_best, dimensions_best = best_empty_area
            new_coordinates.append(coordinates_best)
            # Split the free rectangle
            empty_area_splits = []
            x_best, y_best = coordinates_best
            width_best, height_best = dimensions_best
            empty_area_splits.append(((x_best + box_width, y_best), (width_best - box_width, box_height)))  # Right part
            empty_area_splits.append(((x_best, y_best + box_height), (width_best, height_best - box_height)))  # Bottom part
            # Replace the used free rectangle
            empty_areas.pop(best_empty_area_index)
            for area_split in empty_area_splits:
                if area_split[1][0] > 0 and area_split[1][1] > 0:
                    empty_areas.append(area_split)
        return new_coordinates

    def generate_image(self, images, original_coordinates, texture_dimensions, new_coordinates, image_indices):
        if self.verbose: print("generate_image")
        n = len(original_coordinates)
        # Calculuate the new image size
        max_width = max(new_coordinates[i][0]+texture_dimensions[i][0] for i in range(n))
        max_height = max(new_coordinates[i][1]+texture_dimensions[i][1] for i in range(n))
        new_image_array = np.zeros((max_height, max_width, 4), dtype=np.uint8) # Create a new image array with the calculated size
        # Place textures into the new image
        for i in range(n):
            old_x, old_y = original_coordinates[i]
            width, height = texture_dimensions[i]
            new_x, new_y = new_coordinates[i]
            image_id = image_indices[i]
            new_image_array[new_y:new_y+height, new_x:new_x+width] = images[image_id][old_y:old_y+height, old_x:old_x+width]
        new_image = Image.fromarray(new_image_array) # Convert back to Image
        return new_image

    def process_textures(self, image_paths):
        images = self.load_images(image_paths)
        original_coordinates, texture_dimensions, texture_image_indices = self.identify_bounding_boxes(images)
        packing_bounds = self.calculate_packing_bounds(images)
        new_coordinates = self.pack(packing_bounds, texture_dimensions)
        output_image = self.generate_image(images, original_coordinates, texture_dimensions, new_coordinates, texture_image_indices)
        return output_image