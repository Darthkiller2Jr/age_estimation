from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

import csv
import pandas as pd

from datasets import load_dataset

from time import time

from PIL import Image
import PIL
from PIL import JpegImagePlugin
JpegImagePlugin._getmp = lambda x: None

N1 = 5613
N2 = 7612
M = 400

#for i in range(N):
 #   if  dataset[0].get_format_mimetype() != "image/jpeg" :
  #      print(i)
   #     print(type(dataset[i]))
    #    print(dataset[i])
     #   quit()

#print(dataset[1249].mode)
#quit()
print("")
print("Chargement du modèle...")

processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")

print("Modèle chargé")

#for i in range(N):
 #   print(i)
  #  input = processor(dataset[i], return_tensor = 'pt')

donnees = [["img_name","gender"]]
genders_dict = model.config.id2label

#quit()
img_list = pd.read_csv('appa-real-release/gt_avg_test.csv')['file_name'].to_list()
#print(img_list)
N = len(img_list)
#print(N)
#quit()
for i in range(N//M + 1):
    minimum = i*M
    maximum = min((i+1)*M, N)
    tmp_img_list  = img_list[minimum:maximum]

    dataset = [
        Image.open("appa-real-release/test/"+x).convert('RGB') for x in tmp_img_list
    ]
    t = time()
    print("Processing...")

    inputs = processor(dataset, return_tensors="pt")

    print(f"Processing achieved in {time()-t} s")
    #

    t= time()
    print("Classification ...")

    with torch.no_grad():
        #logits = []
        #for i in range(N):
        #    print(i)
        #    logits.append(model(inputs[i]).logits)

        logits = model(**inputs).logits
    #print(logits)

    predicted_labels = torch.argmax(logits,-1).tolist()

    print(f"Classification achieved in {time()-t}")
    donnees += [[tmp_img_list[i], genders_dict[predicted_labels[i]]] for i in range(maximum-minimum)]


t = time()
print("Constitution du fichier csv...")
# Chemin du fichier CSV à créer
chemin_fichier_csv = "gender_list_test_set.csv"

# Écriture des données dans le fichier CSV
with open(chemin_fichier_csv, mode='a', newline='') as fichier_csv:
    # Création de l'objet writer
    writer = csv.writer(fichier_csv)

    # Écriture des données dans le fichier CSV
    for ligne in donnees:
        writer.writerow(ligne)

print(f"Le fichier CSV '{chemin_fichier_csv}' a été créé avec succès en {time()-t}")