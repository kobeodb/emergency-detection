import argparse
from tkinter import Tk

from src.app import App
from src.main import minio_init, detect


def main():
    client = minio_init()

    win = Tk()
    win.geometry("500x500")
    App(win, client)

    win.mainloop()


if __name__ == '__main__':
    main()
