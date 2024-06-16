import numpy as np
from PIL import Image
import cv2

class Packing_Strategy:
    def __init__(self, progress_info=False) -> None:
        self.use_images_for_packing_bounds = False
        self.progress_info = progress_info

    def calculate_packing_bounds(self, box_dimensions):
        if self.progress_info: print("<F> calculating packing bounds")
        width = max(
            max(box[0] for box in box_dimensions),
            int(np.sqrt(sum([width*height for width,height in box_dimensions])))
            )
        height = None
        return width, height

    def pack(self):
        raise NotImplementedError("Pack not implemented!")


class OG_Strategy(Packing_Strategy):
    def __init__(self, progress_info=True) -> None:
        super().__init__(progress_info)
        self.use_images_for_packing_bounds = True

    def calculate_packing_bounds(self, images):
        if self.progress_info: print("<F> calculating packing bounds")
        width = max(image.shape[1] for image in images)
        height = sum(image.shape[0] for image in images)
        return width, height

    def pack(self, box_dimensions, bounds):
        if self.progress_info:
            print("<F> packing textures")
            print(f"<T> {len(box_dimensions)}")
        new_coordinates = []
        max_width, max_height = bounds
        empty_areas = [((0, 0), (max_width, max_height))]
        for n, box in enumerate(box_dimensions):
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
                if self.progress_info: print(f"<P> {n+1}")
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
            if self.progress_info: print(f"<P> {n+1}")
        return new_coordinates


class NFDH(Packing_Strategy):
    def pack(self, box_dimensions, bounds):
        # Next-Fit Decreasing Height
        if self.progress_info:
            print("<F> packing textures")
            print(f"<T> {len(box_dimensions)}")
        strip_width = bounds[0]
        levels = []
        strip_height = 0
        new_coordinates = []
        for i, (box_width, box_height) in enumerate(box_dimensions):
            packed = False
            for level in levels:
                if not level: continue
                if packed: continue
                if level["remaining"] >= box_width:
                    new_coordinates.append((strip_width-level["remaining"],level["base"]))
                    level["remaining"] -= box_width
                    packed = True
                    if self.progress_info: print(f"<P> {i+1}")
            if not packed:
                new_coordinates.append((0,strip_height))
                if self.progress_info: print(f"<P> {i+1}")
                new_level = {
                    "base": strip_height,
                    "remaining": strip_width - box_width
                }
                levels.append(new_level)
                strip_height += box_height
        return new_coordinates





class Packer:
    def __init__(self, progress_info=False, strategy=NFDH) -> None:
        self.progress_info = progress_info
        # self.strategy = OG_Strategy(progress_info)
        self.strategy = strategy(progress_info)

    def load_images(self, image_paths):
        if self.progress_info:
            print("<F> loading images")
            print(f"<T> {len(image_paths)}")
        image_arrays = []
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            image_array = np.array(image)
            # Add an alpha channel if it doesn't exist
            if image_array.shape[2] == 3:
                alpha_channel = np.ones((image_array.shape[0], image_array.shape[1], 1), dtype=np.uint8) * 255
                image_array = np.concatenate((image_array, alpha_channel), axis=2)
            image_arrays.append(image_array)
            if self.progress_info: print(f"<P> {i+1}")
        return image_arrays

    def identify_textures(self, image_arrays):
        if self.progress_info:
            print("<F> identifying textures")
            print(f"<T> {len(image_arrays)}")
        all_textures = []
        for i, image_array in enumerate(image_arrays):
            # Check if the image is fully opaque and treat the entire image as a single texture if it is
            if np.all(image_array[:, :, 3] == 255):
                textures = [((0, 0), (image_array.shape[1], image_array.shape[0]),i)]  # ((x-offset, y-offset), (width, height), image index)
            else:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
                _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                textures = [((x,y),(w,h),i) for x,y,w,h in [cv2.boundingRect(contour) for contour in contours]]
                if not textures:
                    textures = [((0, 0), (image_array.shape[1], image_array.shape[0]),i)] # Treat the entire image as a texture
            all_textures.extend(textures)
            if self.progress_info: print(f"<P> {i+1}")
        all_textures = sorted(all_textures, key=lambda x: x[1][1], reverse=True) # Sort textures by height (largest to smallest)
        coordinates, dimensions, image_indices = [_ for _ in list(zip(*all_textures))]
        return coordinates, dimensions, image_indices

    def generate_image(self, images, original_coordinates, texture_dimensions, new_coordinates, image_indices):
        n = len(original_coordinates)
        if self.progress_info:
            print("<F> generating new image")
            print(f"<T> {n}")
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
            if self.progress_info: print(f"<P> {i+1}")
        new_image = Image.fromarray(new_image_array) # Convert back to Image
        return new_image

    def process_textures(self, image_paths):
        images = self.load_images(image_paths)
        original_coordinates, texture_dimensions, texture_image_indices = self.identify_textures(images)
        if self.strategy.use_images_for_packing_bounds:
            packing_bounds = self.strategy.calculate_packing_bounds(images)
        else:
            packing_bounds = self.strategy.calculate_packing_bounds(texture_dimensions)
        new_coordinates = self.strategy.pack(texture_dimensions, packing_bounds)
        output_image = self.generate_image(images, original_coordinates, texture_dimensions, new_coordinates, texture_image_indices)
        if self.progress_info: print("<!> FINISHED")
        return output_image