from tkinter import Tk, Label, Button, filedialog
from packer import Packer

# TODO: Go through and clean up code
# TODO: Prettify the GUI and add some polish (incl. taking over the print outputs)

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