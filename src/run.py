from tkinter import Tk

from src.app import App
from src.data.db.main import MinioBucketWrapper


def main():
    client = MinioBucketWrapper()

    win = Tk()
    win.geometry("480x480")
    App(win, client)

    win.mainloop()


if __name__ == '__main__':
    main()
