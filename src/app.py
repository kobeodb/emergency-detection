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
        self.video_paths = {os.path.basename(video): os.path.join("../local/vids", video) for video in self._get_available_videos()}

        # Initialize UI elements
        self.video_dropdown, self.weight_button, self.detect_button, self.train_button, self.video_label = self._init_ui()

        self.stop_event = threading.Event()
        self.video_path, self.weight_path = None, None

    def _init_ui(self):
        """Setup UI elements."""
        top_frame = tk.Frame(self.master, padx=10, pady=5)
        top_frame.grid(row=0, column=0, sticky="w")
        bot_frame = tk.Frame(self.master, padx=15, pady=5)
        bot_frame.grid(row=1, column=0, sticky="n")

        video_label = Label(bot_frame, relief=tk.SUNKEN)
        video_label.grid(row=0, column=0)

        video_dropdown = self._create_dropdown(top_frame, "Select Video", row=0, col=0)
        weight_button = self._create_button(top_frame, "Browse Weights", self._select_weights, row=0, col=1)
        detect_button = self._create_button(top_frame, "Start Detection", self._start_detection, row=0, col=2)
        train_button = self._create_button(top_frame, "Train Model", self._start_training, row=0, col=3)

        return video_dropdown, weight_button, detect_button, train_button, video_label

    def _create_button(self, frame, text, command, row, col):
        button = Button(frame, text=text, command=command)
        button.grid(row=row, column=col, padx=5)
        return button

    def _create_dropdown(self, frame, label_text, row, col):
        var = StringVar(self.master)
        dropdown = OptionMenu(frame, var, *self.video_paths.keys())
        dropdown.grid(row=row, column=col, padx=5)
        return var

    def _get_available_videos(self):
        """Fetches video files from local directory."""
        video_dir = "../local/vids"
        return [f for f in os.listdir(video_dir) if f.endswith('.mp4') or f.endswith('.avi')]

    def _select_weights(self):
        weight_file = filedialog.askopenfilename(filetypes=[("Weight files", "*.pt")])
        if weight_file:
            self.weight_path = weight_file

    def _start_detection(self):
        self.video_path = self.video_paths.get(self.video_dropdown.get())
        if not self.video_path or not self.weight_path:
            messagebox.showerror("Error", "Please select a video and weight file.")
            return

        self.stop_event.clear()
        threading.Thread(target=detect,
                         args=(self.video_path, self.weight_path, self._display_frame, self.stop_event)).start()

    def _start_training(self):
        if not self.weight_path:
            messagebox.showerror("Error", "Please select a weight file.")
            return

        # Use the correct path for 'data.yaml' in the train_model call
        data_yaml_path = os.path.abspath("../src/data/data.yaml")
        threading.Thread(target=train_model, args=(self.weight_path, data_yaml_path)).start()
        messagebox.showinfo("Training", "Training started. This may take some time.")

    def _display_frame(self, img):
        img_resized = self._resize_image(img)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _resize_image(self, img, target_size=(640, 480)):
        return cv2.resize(img, target_size)

    def stop_detection(self):
        self.stop_event.set()
