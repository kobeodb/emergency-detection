from tkinter import Tk
from src.app import BotBrigadeApp

def main():
    root = Tk()
    app = BotBrigadeApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_detection)
    root.mainloop()

if __name__ == '__main__':
    main()
