import cv2
import streamlit as st
import numpy as np

# Charger le classificateur en cascade pour les visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces_from_image(image, scaleFactor, minNeighbors, color):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    # Dessiner des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image, len(faces)


def app():
    st.title("Détection de Visages avec l'Algorithme Viola-Jones")

    # Instructions pour l'utilisateur
    st.write("""
    ### Instructions :
    1. Téléchargez une image contenant des visages.
    2. Ajustez les paramètres pour la détection des visages en utilisant les curseurs.
    3. Choisissez la couleur des rectangles autour des visages.
    4. Cliquez sur le bouton "Détecter les Visages" pour voir les résultats.
    5. Vous pouvez enregistrer l'image avec les visages détectés sur votre appareil.
    """)

    # Télécharger une image
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Lire l'image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)

        # Paramètres de détection
        st.sidebar.header("Paramètres de Détection")
        scaleFactor = st.sidebar.slider("scaleFactor", 1.1, 2.0, 1.3, 0.1)
        minNeighbors = st.sidebar.slider("minNeighbors", 1, 10, 5)

        # Choisir la couleur du rectangle
        color_hex = st.sidebar.color_picker("Choisissez la couleur du rectangle", "#00FF00")
        color = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))  # Convertir hex en tuple RGB

        # Détecter les visages
        if st.button("Détecter les Visages"):
            result_image, num_faces = detect_faces_from_image(image, scaleFactor, minNeighbors, color)

            # Convertir l'image en un format affichable par Streamlit
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.image(result_image_rgb, caption='Image avec détection de visages.', use_column_width=True)
            st.write(f"Nombre de visages détectés : {num_faces}")

            # Enregistrer l'image avec les visages détectés
            result_image_path = 'detected_faces.jpg'
            cv2.imwrite(result_image_path, result_image)

            # Télécharger l'image avec les visages détectés
            with open(result_image_path, "rb") as file:
                st.download_button(
                    label="Télécharger l'image avec détection de visages",
                    data=file,
                    file_name="detected_faces.jpg",
                    mime="image/jpeg"
                )


if __name__ == "__main__":
    app()
