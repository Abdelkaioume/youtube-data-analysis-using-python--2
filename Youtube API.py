from yt_stats import YTstats
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
from matplotlib import *
import matplotlib.pyplot as plt
import json
import pandas as pd

root = tk.Tk()
root.title("Projet Youtube API")
root.iconbitmap('C:\\Users\\ygriy\\Downloads\\ico.ico')

root.configure(bg='#7d2727')

root.geometry("%dx%d+0+0" % (450, 400))

image = Image.open("C:\\Users\\ygriy\\Desktop\\youtube_image.png")
image = image.resize((450, 50), Image.ANTIALIAS)  ## The (250, 250) is (height, width)
photo = ImageTk.PhotoImage(image)

# put the image on a canvas
cv = tk.Canvas(root, width=450, height=50)
cv.pack()
cv.create_image(0, 0, image=photo, anchor='nw')


def execfile():
    channel_id = e1.get()
    from analysis import run
    an = run(channel_id)
    an.rest()
def myClick():
    myLabel1 = tk.Label(root, text="Channel Info downloaded successfully", bg='#7d2727', fg='white', font=('helvetica', 10, 'bold'))
    myLabel1.pack()
    myLabel1.place(relx=0.5, rely=0.62, anchor='center')
    channel_id = e1.get()
    API_KEY = e2.get()
    from yt_stats import YTstats
    yt = YTstats(API_KEY, channel_id)
    yt.extract_all()
    yt.dump()  # dumps to .json
    myButton1 = tk.Button(root, text="Consult Data", bg='black', fg='white', font=('helvetica', 10, 'bold'), command=execfile)
    myButton1.pack()
    myButton1.place(relx=0.5, rely=0.70, anchor='center')

def Comments():
    from subprocess import call
    call(["python", "Comment_interface.py"])
    print("Yes")

myLabel2 = tk.Label(root, text="Enter the Youtube Channel ID ", bg='#7d2727', fg='white', font=('helvetica', 9, 'bold'))
myLabel2.pack()
myLabel2.place(relx=0.5, rely=0.20, anchor='center')

e1 = tk.Entry(root, width=40, text="channel_ID")
e1.pack()
e1.place(relx=0.5, rely=0.25, anchor='center')

myLabel3 = tk.Label(root, text="Enter your personal Google API", bg='#7d2727', fg='white', font=('helvetica', 9, 'bold'))
myLabel3.pack()
myLabel3.place(relx=0.5, rely=0.35, anchor='center')

e2 = tk.Entry(root, width=40)
e2.pack()
e2.place(relx=0.5, rely=0.40, anchor='center')


myButton = tk.Button(root, text="Get Data of the Channel", bg='black', fg='white', font=('helvetica', 10, 'bold'), command=myClick )
myButton.pack()
myButton.place(relx=0.5, rely=0.50, anchor='center')

myLabel3 = tk.Label(root, text="Looking for Comments ? ", bg='#7d2727', fg='white', font=('Calibri', 10, 'bold'))
myLabel3.pack()
myLabel3.place(relx=0.2, rely=0.84, anchor='center')

myButton2 = tk.Button(root, text="Search Comments by Keyword", bg='#c26c0c', fg='black', font=('Calibri', 10,'bold'), command=Comments )
myButton2.pack()
myButton2.place(relx=0.5, rely=0.90, anchor='center')

root.mainloop()