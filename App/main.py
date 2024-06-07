import customtkinter
from PIL import Image
from tkinter import messagebox,filedialog
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
from imutils.video import VideoStream
import os


customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")
#================================
prototxtPath = os.path.sep.join(["config", "deploy.prototxt"])
weightsPath = os.path.sep.join(["config","res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("config/mask_detector.h5")
#================================
def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				faces.append(face)
				locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)

def Vido():
    vs = VideoStream(src=0).start()
    # loop over the frames from the video stream
    imgBackground = cv2.imread('config/images/background.png')
    while True:
        frame = vs.read()
        imgBackground[162:162+480, 55:55+640] = frame
        
        (locs, preds) = detect_and_predict_mask(imgBackground, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask_prob,) = pred
            label = "No Mask" if mask_prob > 0.5 else "Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, mask_prob * 100)
            cv2.putText(imgBackground, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(imgBackground, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Camra Face mask recognition during the COVID-19 pandemic", imgBackground)        
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

def mask_image(image_path):
	image = cv2.imread(image_path)
	(h, w) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
			prediction = maskNet.predict(face)[0]

			# Determine the label based on the prediction
			if prediction > 0.5:
				label = "No Mask"
				confidence = prediction
			else:
				label = "Mask"
				confidence = 1 - prediction
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			label = "{}: {:.2f}%".format(label, confidence[0] * 100)
			cv2.putText(image, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	# show the output image
	cv2.imshow("Image Face mask recognition during the COVID-19 pandemic", image)
	cv2.waitKey(0)


class Main_Page(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face mask recognition during the COVID-19 pandemic")
        self.config(bg="white")
        self.resizable(False, False)
        self.iconbitmap('config/images/mask.ico')

        def def_back():
            if messagebox.askyesno("رسالة تاكيد", f"هل انت متاكد من الخروج من التطبيق ؟") == True:
                self.destroy()
                exit()
        self.protocol('WM_DELETE_WINDOW', def_back)

        main_frame = customtkinter.CTkFrame(master=self, corner_radius=10, fg_color='transparent')
        main_frame.pack(padx=0, pady=0, ipadx=5, ipady=5, fill="both", expand=True)

        left_frame = customtkinter.CTkFrame(master=main_frame, corner_radius=5)
        left_frame.pack(fill="both", expand=True, side="left")

        bg_img = customtkinter.CTkImage(dark_image=Image.open("config/images/logo.jpg"), size=(500, 500))
        bg_lab = customtkinter.CTkLabel(left_frame, image=bg_img, text="")
        bg_lab.pack(fill="both", side="top", expand=True)

        name = customtkinter.CTkLabel(left_frame, text="اعداد المهندس : مجتبى المياحي", text_color="#EAEAEA", font=("", 25, "bold"))
        name.pack(fill="both", side="top", expand=True)

        right_frame = customtkinter.CTkFrame(master=main_frame, corner_radius=5)
        right_frame.pack(padx=15, pady=14, ipadx=2, ipady=2, fill="both", side="right")

        title = customtkinter.CTkLabel(right_frame, text="اكتشاف قناع الوجه", text_color="#EAEAEA", font=("", 23, "bold"))
        title.pack(padx=0, pady=30, expand=True)

        def def_image():
            file_path = filedialog.askopenfilename(initialdir = "/", title = "اختر صورة", filetypes = (("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
            if file_path:
                try:
                    if isinstance(file_path, str):
                        mask_image(file_path) 
                    else:
                        messagebox.showerror('حدث خطأ',"تم تلقي مسار غير صالح. الرجاء تحديد ملف صورة صالح")
                except Exception as e:
                     messagebox.showerror("حدث خطا",f"الخطا هو \n{e}")
            else:
                messagebox.showwarning('حدث خطأ', 'يرجى تحديد الصورة')
        
        icon_image = customtkinter.CTkImage(Image.open(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config"), "images/camera.png")), size=(20, 20))
        button1 = customtkinter.CTkButton(master=right_frame, compound="right", image=icon_image,text="الصورة", corner_radius=7,font=customtkinter.CTkFont(family="Robot", size=15, weight="bold"),command=def_image)
        button1.pack(padx=20, pady=30, expand=True)

        def def_camra():
            Vido()

        icon_video = customtkinter.CTkImage(Image.open(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config"), "images/video-camera-alt.png")), size=(20, 20))
        button2 = customtkinter.CTkButton(master=right_frame, compound="right",image=icon_video, text="كامرا", corner_radius=7,font=customtkinter.CTkFont(family="Robot", size=16, weight="bold"),command=def_camra)
        button2.pack(padx=20, pady=30, expand=True)
        
        button4 = customtkinter.CTkLabel(master=right_frame, text="", font=customtkinter.CTkFont(family="Robot", size=18, weight="bold"))
        button4.pack(padx=20, pady=30, expand=True)

        icon_exit = customtkinter.CTkImage(Image.open(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config"), "images/logout.png")), size=(20, 20))
        button5 = customtkinter.CTkButton(master=right_frame, compound="right",fg_color='#cc0a0a',image=icon_exit,text="الخروج", corner_radius=7,font=customtkinter.CTkFont(family="Robot", size=15, weight="bold"),command=def_back)
        button5.pack(padx=20, pady=30, expand=True)


if __name__ == '__main__':
    app_user = Main_Page()
    app_user.mainloop()