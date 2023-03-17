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
GROUP_NAMES = {"BRPE" : "Brown Pelican", "LAGU" : "Laughing Gull", 
               "BLSK" : "Black Skimmer", "MTRN" : "Mixed Tern",
               "LGHT" : "Large Light Bird", "DARK" : "Large Dark Bird",
               "OTHR" : "Other", "TRSH": "Trash"}
