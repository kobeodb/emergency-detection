import threading
import tkinter as tk
from tkinter import Label, StringVar, OptionMenu, Button, messagebox, filedialog
from PIL import ImageTk, Image
import cv2
import os
from src.main import detect, train_model


class BotBrigadeApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Bot Brigade Detection & Training")

        # Initialize video paths before calling UI setup
        self.video_paths = self._get_available_videos()

        # Initialize UI elements
        self._init_ui()

        self.stop_event = threading.Event()
        self.video_path = None
        self.weight_path = None

    def _init_ui(self):
        """Setup UI elements."""
        top_frame = tk.Frame(self.master, padx=10, pady=5)
        top_frame.grid(row=0, column=0, sticky="w")
        bot_frame = tk.Frame(self.master, padx=15, pady=5)
        bot_frame.grid(row=1, column=0, sticky="n")

        self.video_label = Label(bot_frame, relief=tk.SUNKEN)
        self.video_label.grid(row=0, column=0)

        self.video_var = StringVar(self.master)
        self.video_var.set("Select Video")
        video_dropdown = OptionMenu(top_frame, self.video_var, *self.video_paths.keys())
        video_dropdown.grid(row=0, column=0, padx=5)

        self._create_button(top_frame, "Browse Weights", self._select_weights, row=0, col=1)
        self._create_button(top_frame, "Start Detection", self._start_detection, row=0, col=2)
        self._create_button(top_frame, "Train Model", self._start_training, row=0, col=3)

    def _create_button(self, frame, text, command, row, col):
        """Create a button and place it in the grid."""
        button = Button(frame, text=text, command=command)
        button.grid(row=row, column=col, padx=5)

    def _get_available_videos(self):
        """Fetch video files from local directory."""
        video_dir = "../local/vids"
        return {f: os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))}

    def _select_weights(self):
        """Open file dialog to select YOLO weights file."""
        weight_file = filedialog.askopenfilename(filetypes=[("Weight files", "*.pt")])
        if weight_file:
            self.weight_path = weight_file

    def _start_detection(self):
        """Start video detection in a separate thread."""
        self.video_path = self.video_paths.get(self.video_var.get())
        if not self.video_path or not self.weight_path:
            messagebox.showerror("Error", "Please select a video and weight file.")
            return

        self.stop_event.clear()
        threading.Thread(target=detect, args=(self.video_path, self.weight_path, self._display_frame, self.stop_event)).start()

    def _start_training(self):
        """Start training the YOLO model in a separate thread."""
        if not self.weight_path:
            messagebox.showerror("Error", "Please select a weight file.")
            return

        data_yaml_path = os.path.abspath("../src/data/data.yaml")
        threading.Thread(target=train_model, args=(self.weight_path, data_yaml_path)).start()
        messagebox.showinfo("Training", "Training started. This may take some time.")

    def _display_frame(self, img):
        """Display video frame in the UI."""
        img_resized = self._resize_image(img)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _resize_image(self, img, target_size=(640, 480)):
        """Resize image to target size."""
        return cv2.resize(img, target_size)

    def stop_detection(self):
        """Stop the detection process."""
        self.stop_event.set()


if __name__ == "__main__":
    root = tk.Tk()
    app = BotBrigadeApp(root)
    root.mainloop()
