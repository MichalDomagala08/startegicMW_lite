##############
### Audio Preprocessing Script

"""
    This script aimes at transcribing the audio files of the participants
    Maybe I will use it to my advantage in the future to analyse more of the audio files

    Procedure:
    1. Uses Whisperer from OpenAI to get every
"""

####    Get the Paths to data:
import os 
from storyTools import loadStory,splitStoryEntity,transcibeText
expName = "SecondExp"
datName = "data_second"


### Get Subjects and set up proper Directories
workingDir = os.getcwd();
print( f"Current Firectory: {workingDir}")
path =workingDir + "\\Experiment\\" #data path
print( os.getcwd())
savePath = os.path.join(path,"analysis",expName,'Recalls')

### Make Necessery Directories:
os.makedirs(savePath,exist_ok=True)




path = path +"\\data_second"
print( f"Current Directory: {savePath}")
print( f"Data Directory: {path}")
print([f for f in os.listdir(path) if  not os.path.isfile(os.path.join(path, f))])
subjects = [f for f in os.listdir(path) if  not os.path.isfile(os.path.join(path, f))] # get all the filenames ['ANLU2607','KLKR2608','PASZ2608]

### Load Stories and differentiate between entitites

transcibeText(path,savePath,subjects,device="cuda",langCorr=2)
