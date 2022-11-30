import os

for i in range(1, 17):
  os.system("wget https://zenodo.org/record/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio." + str(i) + ".zip")

os.system("wget https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.meta.zip")
