''' 
Constants for the project 
'''
from PIL import Image

# Fixed parameters
SAVE_FIG = True
Image.MAX_IMAGE_PIXELS = 1000000000
DPI = 500
COL_NAMES = ["class_id", "class_name", "x", "y", "width", "height"]
GROUPS = {"BRPE": ["BRPEA", "BRPEC", "BRPEJ", "BRPEW", "BRPEF"],
          "LAGU": ["LAGUA", "LAGUF", "LAGUJ"],
          "BLSK": ["BLSKA", "BLSKF"],
          "MTRN": ["MTRNA", "MTRNF", "AMAVA", "AMOYA"],
          "LGHT": ["GREGA", "GREGC", "GREGF", "SNEGA", "REEGWMA", 
              "WHIBA", "WHIBC", "WHIBN", "WHIBJ", "ROSPA", "TCHEA", "MEGRT",
              "CAEGA", "CAEGF"],
          "DARK": ["GBHEA", "GBHEC", "GBHEJ", "GBHEN", "GBHEE", "GBHEF",
              "REEGA", "REEGF", "REEGC", "TRHEA", "BCNHA", "DCCOA"], 
          "OTHR": ["OTHRA"],
          "TRSH": ["TRASH"],
        }

GROUP_NAMES = {"BRPE": "Brown Pelican", "LAGU": "Laughing Gull", 
               "BLSK": "Black Skimmer", "MTRN": "Mixed Tern",
               "LGHT": "Large Light Bird", "DARK": "Large Dark Bird",
               "OTHR": "Other", "TRSH": "Trash"}

GROUP_LABELS = {"BRPE": 0, "LAGU": 1, "BLSK": 2, "MTRN": 3, 
                "LGHT": 4, "DARK": 5, "OTHR": 6, "TRSH": 7}

SPECIES_LABELS = {"BRPEA": 0, "BRPEC": 1, "BRPEJ": 2, "BRPEW": 3, "BRPEF": 4, # BRPE
                  "LAGUA": 5, "LAGUF": 6, "LAGUJ": 7, # LAGU
                  "BLSKA": 8, "BLSKF": 9, # BLSK
                  "MTRNA": 10, "MTRNF": 11, "AMAVA": 12, "AMOYA": 13, # MTRN
                  "GREGA": 14, "GREGC": 15, "GREGF": 16, "SNEGA": 17,
                  "REEGWMA": 18, "WHIBA": 19, "WHIBC": 20, "WHIBN": 21,
                  "WHIBJ": 22, "ROSPA": 23, "TCHEA": 24, "MEGRT": 25,
                  "CAEGA": 26, "CAEGF": 27, # LGHT
                  "GBHEA": 28, "GBHEC": 29, "GBHEJ": 30, "GBHEN": 31,
                  "GBHEE": 32, "GBHEF": 33, "REEGA": 34, "REEGF": 35,
                  "REEGC": 36, "TRHEA": 37, "BCNHA": 38, "DCCOA": 39, # DARK
                  "OTHRA": 40, # OTHR
                  "TRASH": 41} # TRSH