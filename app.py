import cv2
import tkinter
from tkinter import filedialog as fd
import numpy as np
import PIL.Image, PIL.ImageTk

class App:
    def __init__(self, window, window_title, image_path="assets/95.jpg"):
        self.window = window
        self.window.title(window_title)
        screenWidth= self.window.winfo_screenwidth()               
        screenHeight= self.window.winfo_screenheight()               
        self.window.geometry("%dx%d" % (screenWidth, screenHeight))

        # Load an image using OpenCV
        self.image_path = image_path
        self.cv_img = cv2.imread(self.image_path, 0)
        self.mofied_Img = cv2.imread(self.image_path, 0)
 
         # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.height, self.width = self.cv_img.shape
 
        # create 2 canvases
        self.canvas_original_img = tkinter.Canvas(window, width = screenWidth / 2 - 50, height = screenHeight / 2 - 50)
        self.canvas_original_img.grid(column=0, row=0, rowspan=3, columnspan=2, padx=5, pady=5)

        self.canvas_modified_img = tkinter.Canvas(window, width = screenWidth / 2 - 50, height = screenHeight / 2 - 50)
        self.canvas_modified_img.grid(column=0, row=3, rowspan=3, columnspan=2, padx=5, pady=5)
 
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
 
        # Add a PhotoImage to the Canvas
        self.canvas_original_img.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)


        self.scaler1 = tkinter.Scale(window, from_=0, to=5,length=800, resolution=0.1, tickinterval=10, orient=tkinter.HORIZONTAL)
        self.scaler1.set(1)
        self.scaler1.grid(column=2, row=0, columnspan=2, padx=5, pady=5, sticky=tkinter.E)
        self.scaler2 = tkinter.Scale(window, from_=0, to=100,length=800, tickinterval=10, orient=tkinter.HORIZONTAL)
        self.scaler2.set(50)
        self.scaler2.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky=tkinter.E)

        self.btn_select_img=tkinter.Button(window, text="Select Img", width=50,command=self.select_image)
        self.btn_select_img.grid(column=2, row=2, columnspan=1, padx=5, pady=5)

        self.btn_reset_img=tkinter.Button(window, text="Reset Img", width=50,command=self.reset_image)
        self.btn_reset_img.grid(column=3, row=2, columnspan=1, padx=5, pady=5)

        self.btn_filter=tkinter.Button(window, text="Filter", width=50,command=self.filter)
        self.btn_filter.grid(column=2, row=3, columnspan=2, padx=5, pady=5)

        self.btn_lowpass_filter=tkinter.Button(window, text="Low-pass Filter", width=50,command=self.lowpass_filter)
        self.btn_lowpass_filter.grid(column=2, row=4, columnspan=2, padx=5, pady=5)

        self.btn_highpass_butterworth=tkinter.Button(window, text="High-pass Butterworth", width=50,command=self.highpass_butterworth)
        self.btn_highpass_butterworth.grid(column=2, row=5, columnspan=1, padx=5, pady=5)

        self.btn_highpass_ideal=tkinter.Button(window, text="High-pass Ideal", width=50,command=self.highpass_ideal)
        self.btn_highpass_ideal.grid(column=3, row=5, columnspan=1, padx=5, pady=5)


 
        self.window.mainloop()

    def select_image(self):
        filename = fd.askopenfilename()
        if filename:
            self.image_path = filename
            # Load an image using OpenCV
            self.cv_img = cv2.imread(self.image_path, 0)
            self.mofied_Img = cv2.imread(self.image_path, 0)
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
            self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
            # Add a PhotoImage to the Canvas
            self.canvas_original_img.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def reset_image(self):
        self.mofied_Img = cv2.imread(self.image_path, 0)
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))

        # Add a PhotoImage to the Canvas
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def filter(self):
        #Filtering
        F = np.fft.fft2(self.mofied_Img)
        F = np.fft.fftshift(F)
        M, N = self.mofied_Img.shape
        D0 = self.scaler2.get()

        u = np.arange(0, M)-M/2
        v=np.arange(0, N) - N/2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = np.array(D<=D0, 'float')
        G = H*F

        G = np.fft.ifftshift(G)
        imgOut = np.real(np.fft.ifft2(G))
        self.mofied_Img = imgOut

        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.mofied_Img))
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)
    
    def lowpass_filter(self):
        F = np.fft.fft2(self.mofied_Img)
        n= self.scaler1.get()
        D0= self.scaler2.get()

        F=np.fft.fftshift(F)
        M, N = self.mofied_Img.shape
        u = np.arange(0, M)-M/2
        v = np.arange(0,N)-N/2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U,2) + np.power(V,2))
        H = 1/np.power(1+ (D/D0), (2*n))
        G = H*F
        G = np.fft.ifftshift(G)
        imgOut = np.real(np.fft.ifft2(G))
        self.mofied_Img = imgOut

        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.mofied_Img))
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def highpass_butterworth(self):
        F = np.fft.fft2(self.mofied_Img)
        M, N = self.mofied_Img.shape
        n = self.scaler1.get()
        D0 = self.scaler2.get()
        u = np.arange(0, M, 1)
        v = np.arange(0,N,1)
        idx = (u > M/2)
        u[idx] = u[idx] - M
        idy = (v > N/2)
        v[idy] = v[idy] - N
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = 1/np.power(1 + (D0/(D+1e-10)), 2*n)
        G = H * F
        imgOut = np.real(np.fft.ifft2(G))
        self.mofied_Img = imgOut

        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.mofied_Img))
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def highpass_ideal(self):
        F = np.fft.fft2(self.mofied_Img)
        M, N = self.mofied_Img.shape
        n = self.scaler1.get()
        D0 = self.scaler2.get()
        u = np.arange(0, M, 1)
        v = np.arange(0,N,1)
        idx = (u > M/2)
        u[idx] = u[idx] - M
        idy = (v > N/2)
        v[idy] = v[idy] - N
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = np.double(D > D0)
        print(D)
        print(D0)
        print(H)
        G = H * F
        imgOut = np.real(np.fft.ifft2(G))
        self.mofied_Img = imgOut

        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.mofied_Img))
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def highpass_gaussian(self):
        pass
# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")