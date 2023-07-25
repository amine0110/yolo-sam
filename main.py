import os
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from tkinter import filedialog
import pydicom
import numpy as np
import ttkbootstrap as ttk
from tkinter import messagebox
from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
import dicom2nifti

class ImageFrame:
    def __init__(self, master):
        self.master = master
        self.pan_image = False
        self.start_x = 0
        self.start_y = 0
        self.images = []
        self.image_index = 0
        self.min_val = -200
        self.max_val = 200
        self.zoom_scale = 1
        self.masks_dict = {}

        # Variables for bounding box
        self.draw_bbox = False
        self.start_bbox_x = 0
        self.start_bbox_y = 0
        self.current_bbox = None

        # Add a mapping from class ids to colors
        self.bbox_colors = {0: '#ffc800', 1: '#00ff00', 2: '#0000ff', 3: '#aa1bc4', 4: '#abc234', 5:'#006faa'}
        self.current_class = 0  # Set the initial class

        # Create frame for buttons
        self.button_frame = ttk.Frame(master, width=200)
        self.button_frame.pack(side="left", fill="y")

        # Create buttons
        self.open_image_button = ttk.Button(self.button_frame, text="Open Image", command=self.open_image)
        self.open_image_button.pack(fill="x", pady=(0,10))
        self.open_directory_button = ttk.Button(self.button_frame, text="Open Directory", command=self.open_directory)
        self.open_directory_button.pack(fill="x", pady=(0,10))

        # Add a button to activate "draw bounding box" mode
        self.draw_bbox_button = ttk.Button(self.button_frame, text="Activate Draw", command=self.activate_bbox_mode)
        self.draw_bbox_button.pack(fill="x", pady=(0,20))

        # Creating the delete button
        delete_button = ttk.Button(self.button_frame, text="Delete Annotation", command=self.delete_file)
        delete_button.pack(fill="x", pady=(0,20))

        # Add an inference button
        self.inference_button = ttk.Button(self.button_frame, text="Launch SAM", command=self.start_inference)
        self.inference_button.pack(fill="x", pady=(0,10))

        # Save dicom masks
        self.save_button = tk.Button(self.button_frame, text="Export Segmentation", command=self.save_masks_as_dicom)
        self.save_button.pack(fill="x")

        # Create canvas
        self.canvas = ttk.Canvas(master, width=616, height=616, bg="white")
        self.canvas.pack(side="right", fill="both", expand=True)

        # Create frame for file list
        self.file_frame = ttk.Frame(master, width=300)
        self.file_frame.pack(side="right", fill="y")

        # Create a scrollbar
        self.scrollbar = ttk.Scrollbar(self.file_frame)
        self.scrollbar.pack(side="right", fill="y")

        # Create a listbox
        self.file_list = tk.Listbox(self.file_frame, yscrollcommand=self.scrollbar.set)
        self.file_list.pack(side="left", fill="both")
        self.scrollbar.config(command=self.file_list.yview)

        # Mouse events for zooming and panning
        self.canvas.bind("<Button-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-1>", self.end_bbox)

        # Bind events for drawing bounding box
        self.canvas.bind("<Button-1>", self.start_bbox)
        self.canvas.bind("<B1-Motion>", self.update_bbox)

        # Create a class selector
        self.class_var = tk.StringVar(master)
        self.class_var.set('0')  # Set the initial class
        self.class_selector = tk.OptionMenu(self.button_frame, self.class_var, *self.bbox_colors.keys())
        self.class_selector.pack(fill="x", pady=(10,20))
        self.class_var.trace('w', self.update_current_class)

        self.min_val_slider_label = ttk.Label(self.button_frame, text='MIN')
        self.min_val_slider_label.pack()
        self.min_val_slider = ttk.Scale(self.button_frame,from_=-3000, to=3000, orient=tk.HORIZONTAL, command=self.update_min_val, length=200)
        self.min_val_slider.pack(fill="x", pady=(0,10))

        self.max_val_slider_label = ttk.Label(self.button_frame, text='MAX')
        self.max_val_slider_label.pack()
        self.max_val_slider = ttk.Scale(self.button_frame, from_=-3000, to=3000, orient=tk.HORIZONTAL, command=self.update_max_val)
        self.max_val_slider.pack(fill="x")

        # Create next and previous buttons
        self.next_previous_buttons_frame = ttk.Frame(self.button_frame)
        self.next_previous_buttons_frame.pack(side='bottom')
        self.previous_button = ttk.Button(self.next_previous_buttons_frame, text="Previous Image", command=self.move_to_previous_image)
        self.previous_button.pack(side="left", padx=(10,10))
        self.next_button = ttk.Button(self.next_previous_buttons_frame, text="Next Image", command=self.move_to_next_image)
        self.next_button.pack(side="right", padx=(10,10))

        # Initialize the SAM model and predictor
        sam_checkpoint = "model/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

        


    def update_current_class(self, *args):
        self.current_class = int(self.class_var.get())
    
    def prepare_dicoms(self, dicom_file_data, max_v=None, min_v=None):

        if max_v: HOUNSFIELD_MAX = int(float(max_v))
        else: HOUNSFIELD_MAX = np.max(dicom_file_data)
        if min_v:HOUNSFIELD_MIN = int(float(min_v))
        else: HOUNSFIELD_MIN = np.min(dicom_file_data)

        HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

        dicom_file_data[dicom_file_data < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
        dicom_file_data[dicom_file_data > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
        normalized_image = (dicom_file_data - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE
        uint8_image = np.uint8(normalized_image*255)

        return uint8_image
    
    
    def prepare_dicoms_sam(self, dcm_file, min_v, max_v):
        dicom_file_data = pydicom.dcmread(dcm_file).pixel_array

        HOUNSFIELD_MAX = max_v # np.max(dicom_file_data)
        HOUNSFIELD_MIN = min_v # np.min(dicom_file_data)

        HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

        dicom_file_data[dicom_file_data < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
        dicom_file_data[dicom_file_data > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
        normalized_image = (dicom_file_data - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE
        uint8_image = np.uint8(normalized_image*255)

        opencv_image = cv2.cvtColor(uint8_image, cv2.COLOR_GRAY2BGR)


        return opencv_image
    
    def load_files(self):
        dir_path = os.path.dirname(self.images[0])
        txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
        self.file_list.delete(0, tk.END)
        for file in txt_files:
            self.file_list.insert(tk.END, file)

    
    def update_min_val(self, value):
        self.min_val = int(float(value))
        self.load_image()

    def update_max_val(self, value):
        self.max_val = int(float(value))
        self.load_image()

    def activate_bbox_mode(self):
        self.draw_bbox = not self.draw_bbox
        # Update button text to reflect current mode
        if self.draw_bbox:
            self.draw_bbox_button.config(text="Deactivate Draw")
        else:
            self.draw_bbox_button.config(text="Activate Draw")

    def start_bbox(self, event):
        if self.draw_bbox:
            self.start_bbox_x = self.canvas.canvasx(event.x)
            self.start_bbox_y = self.canvas.canvasy(event.y)
    
    def update_bbox(self, event):
        if self.draw_bbox:
            if self.current_bbox:
                self.canvas.delete(self.current_bbox)
            end_bbox_x = self.canvas.canvasx(event.x)
            end_bbox_y = self.canvas.canvasy(event.y)
            self.current_bbox = self.canvas.create_rectangle(self.start_bbox_x, self.start_bbox_y, end_bbox_x, end_bbox_y, outline=self.bbox_colors[self.current_class], width=3)
    
    def scale_bbox(self, bbox, scale_factor_x, scale_factor_y):
        x1, y1, x2, y2 = bbox
        scaled_bbox = [x1 * scale_factor_x, y1 * scale_factor_y, x2 * scale_factor_x, y2 * scale_factor_y]
        return scaled_bbox

    def write_bbox_to_file(self):
        # Transform the bounding box coordinates back to the original image size
        bbox = self.canvas.bbox(self.current_bbox)
        scale_factor = min(self.canvas.winfo_width() / self.original_image.width, self.canvas.winfo_height() / self.original_image.height)
        original_bbox = [coord / scale_factor for coord in bbox]

        # Get the file name without extension
        base_name = os.path.splitext(self.images[self.image_index])[0]
            
        # Create the corresponding .txt filename
        txt_filename = f'{base_name}.txt'

        # Calculate the image width and height for normalization
        image_width, image_height = self.original_image.size

        # Calculate the normalized bounding box
        normalized_bbox = [
            original_bbox[0] / image_width,  # x_start
            original_bbox[1] / image_height,  # y_start
            original_bbox[2] / image_width,  # x_end
            original_bbox[3] / image_height  # y_end
        ]

        # Write the class and the normalized bounding box coordinates into the file
        with open(txt_filename, 'a') as f:  # Change 'w' to 'a' to append instead of overwriting
            f.write(f'{self.current_class} ' + ' '.join(map(str, normalized_bbox)) + '\n')  # Add a newline at the end

    def open_image(self):
        img_path = filedialog.askopenfilename(filetypes=(("DICOM Files", "*.dcm"), ("All Files", "*.*")))
        if img_path:
            self.images = [img_path]
            self.image_index = 0
            self.load_image()
            self.load_files()

    def open_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.images = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(('.dcm'))]
            self.image_index = 0
            self.load_image()
            self.load_files()

    def load_image(self):
        # Read DICOM file
        dicom = pydicom.dcmread(self.images[self.image_index])
        dicom = self.prepare_dicoms(dicom.pixel_array, min_v=self.min_val, max_v=self.max_val)
        self.image = Image.fromarray(dicom)
        self.original_image = self.image.copy()

        # Convert the image to RGB
        self.image = self.image.convert("RGB")
        self.original_image = self.original_image.convert("RGB")

        # Get the image size
        image_width, image_height = self.image.size
        new_width = image_width
        new_height = image_height

        # Get the canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Scale the image to fit within the canvas while maintaining its aspect ratio
        if image_width > canvas_width or image_height > canvas_height:
            scale_factor = min(canvas_width / image_width, canvas_height / image_height)
            new_width = int(image_width * scale_factor)
            new_height = int(image_height * scale_factor)
            self.image = self.image.resize((new_width, new_height))

        # Resize the canvas to fit the image
        self.canvas.config(width=new_width, height=new_height)

        # If the .txt file exists, draw a bbox on the image
        txt_file = self.images[self.image_index].replace('.dcm', '.txt')
        bboxes = []
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                for line in f.readlines():
                    content = line.strip().split(' ')
                    bbox_norm = [float(val) for val in content[1:]]  # Ignoring the class id (0) here
                    original_bbox = [
                        bbox_norm[0] * image_width,  # x_start
                        bbox_norm[1] * image_height,  # y_start
                        bbox_norm[2] * image_width,  # x_end
                        bbox_norm[3] * image_height  # y_end
                    ]
                    scaled_bbox = self.scale_bbox(original_bbox, new_width / image_width, new_height / image_height)
                    bboxes.append(scaled_bbox)

                    draw = ImageDraw.Draw(self.image)
                    # Now the outline color can be set to a tuple representing RGB values (0-255 each)
                    draw.rectangle(scaled_bbox, outline=self.bbox_colors[int(content[0])], width=3)  # Use the class id to get the corresponding color
                    del draw

        # If the image has masks saved from previous inference, display them
        if self.images[self.image_index] in self.masks_dict:
            masks = self.masks_dict[self.images[self.image_index]]
            self.display_inference_results(self.images[self.image_index], bboxes)
        else:
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)


    def end_bbox(self, event):
        if self.draw_bbox:
            self.write_bbox_to_file()

    def move_to_previous_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.load_image()
            self.load_files()

    def move_to_next_image(self, event=None):
        if self.image_index < len(self.images) - 1:
            self.image_index += 1
            self.load_image()
            self.load_files()

    def start_pan(self, event):
        self.pan_image = True
        self.start_x = event.x
        self.start_y = event.y

    def stop_pan(self, event):
        self.pan_image = False
        self.start_x = None
        self.start_y = None

    def pan_image(self, event):
        if self.pan_image:
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            self.canvas.move("all", dx, dy)
            self.start_x = event.x
            self.start_y = event.y

    def zoom_image(self, event):
        # Get the current mouse position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Make the zoom level dependent on the direction of the mouse wheel
        if event.delta > 0:
            # Zoom in
            self.image = self.original_image.resize((int(self.image.width * 1.1), int(self.image.height * 1.1)))
        else:
            # Zoom out
            self.image = self.original_image.resize((int(self.image.width * 0.9), int(self.image.height * 0.9)))

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.delete("all")
        self.canvas.create_image(x, y, image=self.tk_image)
    
    def delete_file(self):
        try:
            selected_file = self.file_list.get(self.file_list.curselection())  # Here I'm assuming you're using a file_list widget for file selection
            if selected_file.endswith('.txt'):
                response = messagebox.askyesno("Confirmation",
                                            f"Do you really want to delete {selected_file}?")
                if response == 1:  # If user clicks 'yes'
                    os.remove(os.path.join(os.path.dirname(self.images[0]), selected_file) )
                    self.file_list.delete(self.file_list.curselection())  # Remove the file from the file_list
            else:
                messagebox.showerror("Error", "Selected file is not a txt file.")
        
        except:
            print('Problem with the file delete')

    def start_inference(self):
        print('[INFO]: Starting inference')
        # Create a dictionary to store masks for each image
        self.masks_dict = {}

        # Assuming the DICOM files are in self.images and bounding boxes in .txt files
        for image_path in self.images:
            txt_file = image_path.replace('.dcm', '.txt')

            # Continue if there's no associated txt file
            if not os.path.exists(txt_file):
                continue
            
            img = self.prepare_dicoms_sam(image_path, min_v=self.min_val, max_v=self.max_val)
            self.predictor.set_image(img)

            img_width, img_height, _ = img.shape

            # Load all bounding boxes
            bounding_boxes = []
            with open(txt_file, 'r') as f:
                content = f.readlines()
                for line in content:
                    bbox_norm = [float(val) for val in line.strip().split()[1:]]  # Ignoring the class id (0) here
                    bbox = [
                        int(bbox_norm[0] * img_width),  # x_start
                        int(bbox_norm[1] * img_height),  # y_start
                        int(bbox_norm[2] * img_width),  # x_end
                        int(bbox_norm[3] * img_height)  # y_end
                    ]
                    bounding_boxes.append(bbox)
                
            # Convert list of bounding boxes to torch tensor
            bounding_boxes_torch = torch.tensor(bounding_boxes, device=self.predictor.device)

            # Apply transform to bounding boxes
            transformed_boxes = self.predictor.transform.apply_boxes_torch(bounding_boxes_torch, img.shape[:2])

            # Perform prediction for all bounding boxes at once
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            # Save masks for this image
            self.masks_dict[image_path] = masks
            self.display_inference_results(image_path, bounding_boxes_torch)
    
    def display_inference_results(self, image_path, bboxes):
        # Retrieve the saved masks for this image
        masks = self.masks_dict.get(image_path, [])

        # Continue if there are no saved masks
        if masks.nelement() == 0:
            return

        # Load image and prepare for overlaying the masks
        img = self.prepare_dicoms_sam(image_path, min_v=self.min_val, max_v=self.max_val)
        img = Image.fromarray(img)
        img = img.convert("RGB")

        # Create an empty Image object
        img_with_mask = Image.new('RGBA', img.size)

        # Paste the original image
        img_with_mask.paste(img)

        # Ensure the length of masks and bboxes are the same
        assert len(masks) == len(bboxes), "The number of masks should be equal to the number of bounding boxes."

        for mask, bbox in zip(masks, bboxes):
            # Create a mask image
            mask_img = Image.fromarray((mask[0].cpu().numpy() * 255).astype('uint8'), 'L')
            colored_mask = Image.new('RGBA', img.size, color=(255, 200, 0, int(0.6*255)))  # Yellow color with 0.6 alpha
            img_with_mask.paste(colored_mask, mask=mask_img)

            # Draw the bounding box
            draw = ImageDraw.Draw(img_with_mask)
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline='green', width=2)

        # Convert the final image to PhotoImage and display on the canvas
        self.final_image_tk = ImageTk.PhotoImage(img_with_mask)
        self.canvas.create_image(0, 0, image=self.final_image_tk, anchor='nw')
    
    def save_masks_as_dicom(self):
        ask_file = filedialog.askdirectory()
        if ask_file:
            print('[INFO]: Exporting masks to dicom')
            for image_path in self.images:
                dicom = pydicom.dcmread(image_path)
                masks = self.masks_dict.get(image_path)

                if masks is not None:  # If masks are available
                    # Initialize an empty numpy array to store the combined mask
                    combined_mask = np.zeros_like(masks[0][0].cpu().numpy())

                    # Iterate over all masks and combine them
                    for mask in masks:
                        mask_np = mask[0].cpu().numpy()
                        combined_mask = np.maximum(combined_mask, mask_np)

                    # Replace the pixel array with the combined mask
                    dicom.PixelData = (combined_mask * 255).astype(np.uint8).tobytes()
                else:  # If no masks
                    # Create an empty pixel array
                    dicom.PixelData = np.zeros_like(dicom.pixel_array, dtype=np.uint8).tobytes()

                # Set photometric interpretation to "MONOCHROME2"
                dicom.PhotometricInterpretation = "MONOCHROME2"

                # Set pixel representation and bits allocated for grayscale image
                dicom.PixelRepresentation = 0
                dicom.BitsAllocated = 8

                # Save the new DICOM file
                save_path = os.path.join(ask_file, 'dicom_masks')
                os.makedirs(save_path, exist_ok=True)

                dicom.save_as(os.path.join(save_path, os.path.basename(image_path).replace('.dcm', '_seg.dcm')))

            print('[INFO]: Converting dicom to nifti')
            dicom2nifti.dicom_series_to_nifti(save_path, os.path.join(ask_file, f'{os.path.basename(os.path.dirname(self.images[0]))}_sam_segmentation.nii.gz'))
            dicom2nifti.dicom_series_to_nifti(os.path.dirname(self.images[0]), os.path.join(ask_file, f'{os.path.basename(os.path.dirname(self.images[0]))}.nii.gz'))

            print('[INFO]: Files Exported')
            
if __name__ == '__main__':
    ttk_root = ttk.Window()
    ttk_root.style.theme_use('superhero') 
    ttk_root.title('Medical Annotation')
    ttk_root.iconbitmap('utils/icon.ico')

    app = ImageFrame(ttk_root)

    ttk_root.mainloop()

