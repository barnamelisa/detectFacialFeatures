import cv2
import dlib
import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk

# Inițializează detectorul facial și predictorul de puncte caracteristice
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Funcția de detectare a punctelor caracteristice ale feței din imagine
def detect_facial_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # """  Conversia în Nuante de Gri """

   # gray = gray / 255.0 # """ Normalizare """

    gray = cv2.GaussianBlur(gray, (5, 5), 0) # """ Filtru Gaussian """, unde 255 este val max a unui pixel

    faces = detector(gray) # """ HOG (Histogram of Oriented Gradients) """

    for face in faces:
        landmarks = predictor(gray, face)  # """ Shape Predictor """
        for n in range(36, 48):  # Ochi
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        for n in range(48, 68):  # Gură
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        for n in range(27, 36):  # Nas
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image


# Funcția pentru încărcarea și afișarea imaginii într-o fereastră nouă
def analyze_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        print(f"Imagine selectată: {file_path}")  # Debug: afisează calea fișierului
        img = cv2.imread(file_path)

        if img is None:
            print("Eroare: Imagina nu a fost încărcată corect!")
            return  # Dacă nu s-a încărcat imaginea, ieșim din funcție

        # Procesăm imaginea
        h, w = img.shape[:2]
        max_size = 480
        if h > w:
            ratio = max_size / h
            new_size = (int(w * ratio), max_size)
        else:
            ratio = max_size / w
            new_size = (max_size, int(h * ratio))

        img = cv2.resize(img, new_size)  # Redimensionarea imaginii
        img = detect_facial_features(img)  # Detectarea caracteristicilor faciale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Conversia la RGB

        # Conversia la format compatibil Tkinter
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Creează o fereastră nouă pentru afișarea imaginii
        new_window = Toplevel(root)
        new_window.title("Imagine Procesată")
        new_window.geometry("600x500")
        new_window.configure(bg="#f0f4f7")

        # Afișează imaginea în noua fereastră
        panel = tk.Label(new_window, image=img_tk, bg="#f0f4f7")
        panel.image = img_tk  # Stocăm referința pentru a preveni ștergerea din memorie
        panel.pack(pady=10)

        # Buton de "Înapoi" pentru a închide fereastra secundară
        back_button = tk.Button(new_window, text="Înapoi", command=new_window.destroy, font=("Arial", 12), bg="#4a90e2", fg="white")
        back_button.pack(pady=10)



# Funcția pentru analiza morfologică în timp real
def real_time_analysis():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_facial_features(frame)
        cv2.imshow("Analiza Morfologică în Timp Real", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Apasă 'q' pentru a închide
            break

    cap.release()
    cv2.destroyAllWindows()


# Setează interfața grafică folosind Tkinter
root = tk.Tk()
root.title("Detectare Componente Faciale")

# Setează dimensiunea ferestrei
root.geometry("500x350")
root.minsize(500, 350)

# Setează icon-ul ferestrei (înlocuiește "icon.ico" cu numele fișierului tău de icon)
root.iconbitmap("face-recognition.ico")

# Setează culoarea de fundal
root.configure(bg="#f0f4f7")

# Creează un container pentru centrare
frame_center = tk.Frame(root, bg="#f0f4f7")
frame_center.pack(expand=True)

# Creează butoane pentru funcționalitățile aplicației, centralizate
btn_image_analysis = tk.Button(
    frame_center,
    text="Analiză Morfologică din Fotografii",
    command=analyze_image,
    font=("Arial", 12),
    width=30,
    height=2,
    bg="#4a90e2",
    fg="white",
    activebackground="#357ABD",
    activeforeground="white"
)
btn_image_analysis.pack(pady=10)

btn_real_time_analysis = tk.Button(
    frame_center,
    text="Analiză Morfologică în Timp Real",
    command=real_time_analysis,
    font=("Arial", 12),
    width=30,
    height=2,
    bg="#4a90e2",
    fg="white",
    activebackground="#357ABD",
    activeforeground="white"
)
btn_real_time_analysis.pack(pady=10)

# Rulează interfața grafică
root.mainloop()
