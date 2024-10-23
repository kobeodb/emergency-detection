import os
import threading
import tkinter as tk
from tkinter import Label, StringVar, OptionMenu, Button, messagebox, filedialog

import cv2
from PIL import ImageTk, Image

from src.main import detect


class App:
    def __init__(self, master, client):
        self.master = master
        self.client = client

        self.master.title("Bot Brigade")

        self.top_row = tk.Frame(self.master)
        self.top_row.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.bot_row = tk.Frame(self.master)
        self.bot_row.grid(row=1, column=0, padx=15, pady=5, sticky="n")

        self.available_videos = self.get_available_videos()

        self.video_paths = {os.path.basename(video): video for video in self.available_videos}

        self.selected_video_name = StringVar(master)

        if self.available_videos:
            self.selected_video_name.set(list(self.video_paths.keys())[0])
        else:
            self.selected_video_name.set("No videos available")

        self.video_dropdown_label = OptionMenu(self.top_row, self.selected_video_name, *sorted(self.video_paths.keys()))
        self.video_dropdown_label.grid(row=0, column=0, padx=5, sticky="w")

        self.weight_button_text = StringVar()
        self.weight_button_text.set("Weights")

        self.weight_browse_button = Button(self.top_row, textvariable=self.weight_button_text, command=self.browse_weight_file)
        self.weight_browse_button.grid(row=0, column=1, padx=5, sticky="w")

        self.start_button = Button(self.top_row, text="Detect", command=self.start_detection)
        self.start_button.grid(row=0, column=2, padx=5, sticky="w")

        self.stop_button = Button(self.top_row, text="Stop", command=self.stop_detection)
        self.stop_button.grid(row=0, column=3, padx=5, sticky="w")

        self.video_frame_width = 640
        self.video_frame_height = 480

        self.video_label = Label(self.bot_row, relief=tk.SUNKEN)
        self.video_label.grid(row=0, column=0)

        self.video_frame = tk.Frame(self.video_label, width=self.video_frame_width, height=self.video_frame_height)
        self.video_frame.grid(row=0, column=0)

        self.img_tk = None
        self.weight_path = None
        self.video_path = None

        self.stop = threading.Event()
        self.master.protocol("WM_DELETE_WINDOW", self.cleanup)

    def get_available_videos(self):
        try:
            return [obj for obj in self.client.list_obj()
                    if obj.endswith('.mp4')
                    or obj.endswith('.avi')]

        except Exception as e:
            messagebox.showerror("Error", f"Could not fetch video files from MinIO: {str(e)}")
            return []

    def browse_weight_file(self):
        weight_file = filedialog.askopenfilename(filetypes=[("Weight files", "*.pt")])
        if weight_file:
            self.weight_button_text.set(os.path.basename(weight_file))
            self.weight_path = weight_file

    def start_detection(self):
        self.video_path = self.video_paths[self.selected_video_name.get()]

        if not self.video_path:
            messagebox.showerror("Error", "No valid video file selected.")
            return

        if not self.weight_path:
            messagebox.showerror("Error", "No valid weight file selected.")
            return

        self.stop.clear()
        try:
            threading.Thread(target=detect, args=(
                self.client,
                self.video_path,
                self.weight_path,
                self.update,
                self.stop)
            ).start()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during detection: {str(e)}")

    def stop_detection(self):
        self.stop.set()
        self.video_label.configure(image='')
        self.video_label.imgtk = None

    def update(self, img):
        # Resize image to fit in the predefined box
        img = self.resize_image_to_fit_box(img)

        cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.master.after(0, self._update_image, imgtk)

    def resize_image_to_fit_box(self, img):
        h, w = img.shape[:2]

        scale = min(self.video_frame_width / w, self.video_frame_height / h)

        new_width = int(w * scale)
        new_height = int(h * scale)

        return cv2.resize(img, (new_width, new_height))

    def _update_image(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def cleanup(self):
        self.stop_detection()
        self.master.destroy()
