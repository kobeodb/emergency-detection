import os
import threading
from tkinter import Label, StringVar, OptionMenu, Button, messagebox, filedialog
import tkinter as tk

import cv2
from PIL import ImageTk, Image
from src.main import detect


class App:
    def __init__(self, master, client):
        self.master = master
        self.client = client

        self.master.title("Bot Brigade")

        self.label = Label(master, text="Select a video for detection")
        self.label.pack()

        self.available_videos = self.get_available_videos()

        self.video_path = StringVar(master)
        if self.available_videos:
            self.video_path.set(self.available_videos[0])
        else:
            self.video_path.set("No videos available")

        self.video_dropdown = OptionMenu(master, self.video_path, *self.available_videos)
        self.video_dropdown.pack(padx=5)

        self.label = Label(master, text="Select a weight file for detection")
        self.label.pack()

        self.available_weights = self.get_available_weights()

        self.weight_path = StringVar(master)
        if self.available_weights:
            self.weight_path.set(self.available_weights[0])
        else:
            self.weight_path.set("No weight files found")

        # Weight dropdown and file browser
        self.weight_dropdown_frame = tk.Frame(master)
        self.weight_dropdown_frame.pack(pady=5)

        self.weight_dropdown = OptionMenu(self.weight_dropdown_frame, self.weight_path, *self.available_weights)
        self.weight_dropdown.pack(side=tk.LEFT, padx=5)

        self.weight_browse_button = Button(self.weight_dropdown_frame, text="Browse", command=self.browse_weight_file)
        self.weight_browse_button.pack(side=tk.LEFT)

        self.video_label = Label(master)
        self.video_label.pack()

        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=5)

        self.start_button = Button(self.button_frame, text="Start", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.quit_button = Button(self.button_frame, text="Quit", command=master.quit)
        self.quit_button.pack(side=tk.LEFT, padx=5)

        self.img_tk = None

    def get_available_weights(self):
        try:
            path = './data/weights/'
            return [path + w for w in os.listdir(path) if w.endswith('.pt')]
        except Exception as e:
            messagebox.showerror("Error", f"Could not fetch weight files from directory: {str(e)}")

    def get_available_videos(self):
        try:
            return [obj for obj in self.client.list_obj()
                    if obj.endswith('.mp4')
                    or obj.endswith('.avi')]

        except Exception as e:
            messagebox.showerror("Error", f"Could not fetch video files from MinIO: {str(e)}")
            return []

    def browse_video_file(self):
        video_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if video_file:
            self.video_path.set(video_file)

    def browse_weight_file(self):
        weight_file = filedialog.askopenfilename(filetypes=[("PyTorch Model files", "*.pt")])
        if weight_file:
            self.weight_path.set(weight_file)

    def start_detection(self):
        video = self.video_path.get()
        if video == "No videos available" and not os.path.isfile(video):
            messagebox.showerror("Error", "No valid video file selected.")
            return

        weight = self.weight_path.get()
        if weight == "No weight files found" and not os.path.isfile(weight):
            messagebox.showerror("Error", "No valid weight file selected.")
            return

        try:
            threading.Thread(target=detect, args=(self.client, video, weight, self.update)).start()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during detection: {str(e)}")

    def update(self, img):
        cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(cv2img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
