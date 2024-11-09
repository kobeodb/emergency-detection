import threading
import tkinter as tk
from tkinter import Label, Button, messagebox, filedialog, Frame, OptionMenu, StringVar
from PIL import ImageTk, Image
import cv2
import os
from src.fall_detector.train_fall_detector import detect, train_model


class BotBrigadeApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Bot Brigade Detection & Training")
        self.master.geometry("1200x800")
        self.master.configure(bg='#f0f0f0')

        # Initialize video paths before calling UI setup
        self.video_paths = self._get_available_videos()

        # Initialize UI elements
        self._init_ui()

        self.stop_event = threading.Event()
        self.video_path = None
        self.weight_path = None

    def _init_ui(self):
        """Setup UI elements."""
        top_frame = Frame(self.master, padx=10, pady=10, bg='#f0f0f0')
        top_frame.pack(fill=tk.X, side=tk.TOP)
        bot_frame = Frame(self.master, padx=10, pady=10, bg='#f0f0f0')
        bot_frame.pack(fill=tk.BOTH, expand=True)

        # Video Display Label
        main_video_frame = Frame(bot_frame, bg='#f0f0f0')
        main_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = Label(main_video_frame, relief=tk.SUNKEN, width=640, height=480, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Video Selection Dropdown
        self.video_var = StringVar(self.master)
        self.video_var.set("Select a video")
        self.video_dropdown = OptionMenu(top_frame, self.video_var, *self.video_paths.keys(),
                                         command=self._select_video_dropdown)
        self.video_dropdown.config(width=20, padx=10, pady=5, bg='#4CAF50', fg='black')
        self.video_dropdown.pack(side=tk.LEFT, padx=5)

        # Browse Weights Button
        self._create_button(top_frame, "Browse Weights", self._select_weights, side=tk.LEFT)
        # Start Detection Button
        self._create_button(top_frame, "Start Detection", self._start_detection, side=tk.LEFT)
        # Train Model Button
        self._create_button(top_frame, "Train Model", self._start_training, side=tk.LEFT)

        # Weight File Label
        self.weight_label = Label(top_frame, text="No weight file selected", bg='#f0f0f0', fg='black')
        self.weight_label.pack(side=tk.LEFT, padx=10)

    def _create_button(self, frame, text, command, side):
        """Create a button and place it in the grid."""
        button = Button(frame, text=text, command=command, padx=10, pady=5, bg='#4CAF50', fg='black')
        button.pack(side=side, padx=5)

    def _get_available_videos(self):
        """Fetch video files from local directory."""
        video_dir = "../../local/vids"
        return {f: os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))}

    def _select_video_dropdown(self, video_name):
        """Select a video from dropdown and show a preview."""
        video_path = self.video_paths.get(video_name)
        if video_path:
            self.video_path = video_path
            self._show_video_preview(video_path)

    def _select_weights(self):
        """Open file dialog to select YOLO weights file."""
        weight_file = filedialog.askopenfilename(filetypes=[("Weight files", "*.pt")])
        if weight_file:
            self.weight_path = weight_file
            self.weight_label.config(text=f"Selected: {os.path.basename(weight_file)}")

    def _start_detection(self):
        """Start video detection in a separate thread."""
        if not self.video_path or not self.weight_path:
            messagebox.showerror("Error", "Please select a video and weight file.")
            return

        self.stop_event.clear()
        threading.Thread(target=detect,
                         args=(self.video_path, self.weight_path, self._display_frame, self.stop_event)).start()

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

    def _show_video_preview(self, video_path):
        """Show a preview of the selected video in the label."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            img_resized = self._resize_image(frame, target_size=(320, 240))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        cap.release()

    def stop_detection(self):
        """Stop the detection process."""
        self.stop_event.set()



def main():
    root = tk.Tk()
    app = BotBrigadeApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_detection)
    root.mainloop()

if __name__ == '__main__':
    main()