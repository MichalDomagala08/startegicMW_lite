import pygame
from pygame.locals import * 
import pandas as pd
import pyaudio, wave
import pylink
import time
from tests.testFunctions import testTiming,compareEyeTrackingWithBeh,prepareEyeTrackingTiming
from classes.fileIDInput import createOutputs
from classes.CalibrationGraphivs import CalibrationGraphics
from classes.Audio import audioTrial
from classes.Audio import storyTimeDict3
from classes.recallTrial import recallTrial
from classes.welcomeMessage import welcomeMessage
from classes.welcomeMessage import generateMessages
from  random import uniform

import pickle
import sys,os,subprocess

#### ----- Setups ------ ####

#Getting User Info:
outputControll = createOutputs(dataPath=".\\Experiment\\data");
outputControll.idInputwin()
outputControll.write("\n       ###### STRATEGIC MW EXPERIMENT v0.98 ######    \n")

outputControll.write(f"  Currently Processing Subject N.: {outputControll.ID}")
if outputControll.ID == "" or len(outputControll.ID) < 3:
     raise ValueError("Invalid ID - Exiting PRocedure\n\n") # To Do XD 

# Display Information
pygame.init()
disp = pylink.getDisplayInformation()
outputControll.write("  Current Display Information: ")
outputControll.write(f"      Width:  {disp.width}")
outputControll.write(f"      Height: {disp.height}\n")

width = disp.width
height =disp.height
screen = pygame.display.set_mode((width, height))
tempInitialTime = 0; # For Gathering Realtive Timestamps
# PyGame setups
font = pygame.font.Font(None, 50)
pygame.mouse.set_visible(False)
pygame.display.set_caption("Strategic MW Experiment")

# story Global parameters Setup:
storyPart = "welcome1"  # Controlling the experiment flow
dummyMode = True
TestMode = False
SCREEN_WIDTH_CM = 53 #Width
SCREEN_HEIGHT_CM = 30 # Height 
VIEWING_DISTANCE_CM = 93 # 
entityName = [storyTimeDict3["partNames"][0][:-2], storyTimeDict3["partNames"][1][:-2]][int(uniform(0, 1))]
outputControll.write(f"      Viewing Distance:  {VIEWING_DISTANCE_CM}\n")

outputControll.write("  Experiment Config: ")
outputControll.write(f"      dummyMode:             {dummyMode}")
outputControll.write(f"      Current Named Entity:  {entityName}")
outputControll.write(f"      Test Mode:             {TestMode}")

### Make Experiment Objects

# Experiment  Message Initialization
[WelcomeMessage1, WelcomeMessage11, WelcomeMessage2, WelcomeMessage3,WelcomeMessage21alt,exitMessage1] = generateMessages(entityName)
                                            # Message dictionary                               Font, Screen, Next exp Part, prev exp Part
welcome = welcomeMessage([WelcomeMessage1,WelcomeMessage11, WelcomeMessage2,  WelcomeMessage3],font,screen,"welcome1","calibration1")
welcome2 = welcomeMessage([WelcomeMessage21alt],font,screen,"welcome2","story1")
calib2 = welcomeMessage(WelcomeMessage3,font,screen,"calib_text","calibration2")
exitMessage = welcomeMessage([exitMessage1],font,screen,"exit","")

# Audio File initialization:
Story1 = audioTrial(r".\TextToSpeech\Story3_AIpart1",storyTimeDict3,font,screen,"story1","calib_text",outputControll,verbose=2)
Story2 = audioTrial(r".\TextToSpeech\Story3_AIpart2",storyTimeDict3,font,screen,"story2","recall1",outputControll,verbose=2)

# Audio Recording Object Initialization:
recall1 = recallTrial("story1.wav",font,screen,"recall1","exit",outputControll)

# connect to the tracker
if not dummyMode:
    el_tracker = pylink.EyeLink('100.1.1.1')

    # open an EDF data file on the Host PC
    outputControll.manageEyeTrackingFile(el_tracker)

    # send over a command to let the tracker know the correct screen resolution
    scn_coords = "screen_pixel_coords = 0 0 %d %d" % (width - 1, height - 1)
    el_tracker.sendCommand(scn_coords)
    el_tracker.sendCommand(f"screen_phys_coords = -{SCREEN_WIDTH_CM/2*10} {SCREEN_HEIGHT_CM/2*10} {SCREEN_WIDTH_CM/2*10} -{SCREEN_HEIGHT_CM/2*10}")

    # Send viewing distance (in millimeters)
    el_tracker.sendCommand(f"screen_distance = {VIEWING_DISTANCE_CM*10}")

    # Instantiate a graphics environment (genv) for calibration
    genv = CalibrationGraphics(el_tracker, screen)
else:
    el_tracker = pylink.EyeLink(None)

outputControll.write("\nEXPERIMENTS STARTS")



### ------ Experiment ------- ####
el_tracker.sendCommand("file_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS")
el_tracker.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON")
el_tracker.sendCommand("pupil_size_diameter = YES")
running = True
begBlockFlag = True
while running: 
    screen.fill((127, 127, 127))  # Clear screen before each frame
    
    if storyPart == "welcome1":
       if begBlockFlag:
           begBlockFlag = False

       storyPart = welcome.run()

       if storyPart != "welcome1":
            begBlockFlag = True
    elif storyPart == "calibration1":
        tempInitialTime = time.time();
        outputControll.write(f"\nCalibration of EyeTracker ({time.time():.3f})\n")

        if not dummyMode:
            try:
                pylink.pumpDelay(50)
                pylink.openGraphicsEx(genv)  # Register CalibrationGraphics

                el_tracker.sendCommand("automatic_calibration_pacing = 1000")  # Pacing of Targets - allow for automaticity
                el_tracker.doTrackerSetup()  # Calibration Setup 

                pylink.pumpDelay(50)            
            except RuntimeError as err:
                print('ERROR:', err)
                el_tracker.exitCalibration()     

        outputControll.write(f"   Calibration Ended (Duration: {time.time()-tempInitialTime})")

        storyPart = "welcome2"
        begBlockFlag = True

        screen.fill((255/2, 255/2, 255/2))  # Reset screen
        pygame.display.flip()  # Ensure Pygame updates after calibration
           
    elif storyPart == "welcome2":
       
        if begBlockFlag:
           begBlockFlag = False

        storyPart = welcome2.run()

        if storyPart != "welcome2":
           begBlockFlag = True
        
    elif storyPart == "story1":
        #insert my AudioClass.run() Here!!!
        if begBlockFlag:
            el_tracker.startRecording(1, 1, 1, 1)
            pylink.pumpDelay(100)  # Small delay to ensure recording starts

            outputControll.write(f"\nStarting Audio Trial 1 ({time.time():.3f}) *** ")
            outputControll.writeToEyeLink("\tLISTEN\tSTORY_1\tBEG")
            begBlockFlag = False

        storyPart = Story1.run()
        if storyPart != "story1":
            outputControll.writeToEyeLink("\tLISTEN\tSTORY_1\tEND")
            outputControll.write("\nAnswer Timing Log:")
            outputControll.write(pd.DataFrame(Story1.timingLog,columns=["Event","StoryTime","RT","partDuration"]).to_string())
            begBlockFlag = True
            testTiming(Story1.timingLog,outputControll)

    elif storyPart == "calib_text":
       
        if begBlockFlag:
           begBlockFlag = False

        storyPart = welcome2.run()

        if storyPart != "calibration2":
           begBlockFlag = True
        
    elif storyPart == "calibration2":
        tempInitialTime = time.time();
        outputControll.write(f"\nCalibration of EyeTracker ({time.time():.3f})\n")

        if not dummyMode:
            try:
                pylink.pumpDelay(50)
                pylink.openGraphicsEx(genv)  # Register CalibrationGraphics

                el_tracker.sendCommand("automatic_calibration_pacing = 1000")  # Pacing of Targets - allow for automaticity
                el_tracker.doTrackerSetup()  # Calibration Setup 

                pylink.pumpDelay(50)            
            except RuntimeError as err:
                print('ERROR:', err)
                el_tracker.exitCalibration()     

        outputControll.write(f"   Calibration Ended (Duration: {time.time()-tempInitialTime})")

        storyPart = "story2"
        begBlockFlag = True

        screen.fill((255/2, 255/2, 255/2))  # Reset screen
        pygame.display.flip()  # Ensure Pygame updates after calibration

    elif storyPart == "story1":
        #insert my AudioClass.run() Here!!!
        if begBlockFlag:
            el_tracker.startRecording(1, 1, 1, 1)
            pylink.pumpDelay(100)  # Small delay to ensure recording starts

            outputControll.write(f"\nStarting Audio Trial 1 ({time.time():.3f}) *** ")
            outputControll.writeToEyeLink("\tLISTEN\tSTORY_1\tBEG")
            begBlockFlag = False

        storyPart = Story1.run()
        if storyPart != "story1":
            outputControll.writeToEyeLink("\tLISTEN\tSTORY_1\tEND")
            outputControll.write("\nAnswer Timing Log:")
            outputControll.write(pd.DataFrame(Story1.timingLog,columns=["Event","StoryTime","RT","partDuration"]).to_string())
            begBlockFlag = True
            testTiming(Story1.timingLog,outputControll)

    elif storyPart == "recall1":
        if begBlockFlag:
            el_tracker.stopRecording()

            outputControll.write(f"\nRecall of Audio  1 ({time.time():.3f} *** \n")

            outputControll.writeToEyeLink("\tRECALL\tSTORY_1\tBEG")
            begBlockFlag = False

        storyPart = recall1.run()

        if storyPart != "recall1":
            outputControll.writeToEyeLink("\tRECALL\tSTORY_1\tEND")
            begBlockFlag = True
    elif storyPart == "exit":
        if begBlockFlag:
            outputControll.write(f"\nEND OF A PROCEDURE ({time.time():.3f}) *** \n")

            outputControll.writeToEyeLink("\tPROCEDURE\t0\tEND")

            begBlockFlag = False

        storyPart = exitMessage.run()

        if storyPart != "exit":
           begBlockFlag = True  
    else:
        running = False

    pygame.display.flip()  # Update display

#### --- Saving and Tests --- ####
if not dummyMode:
    edf_path = os.path.abspath(os.path.join(outputControll.dataPath, outputControll.ID))
    if not os.path.exists(edf_path):
        os.makedirs(edf_path)  # 

    # Save Data to a pickle
    logs = {'story1_beh': pd.DataFrame(Story1.timingLog,columns=["Event","StoryTime","RT","partDuration"])}
    with open(os.path.join(edf_path,f'{outputControll.ID}.pickle'), 'wb') as handle:
        pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filePath = os.path.join(edf_path,f'{outputControll.ID}.edf');
    el_tracker.closeDataFile()  # Close the EDF file
    el_tracker.receiveDataFile(f'{outputControll.ID}.edf',os.path.join(edf_path,f'{outputControll.ID}.edf'))
    outputControll.write(f"Eye Tracking data saved to: {filePath}")
    asciiFilePath = outputControll.edf2ascii(filePath);

    ### TESTS
    EyetrackerDF = prepareEyeTrackingTiming(os.path.join(edf_path,f'{outputControll.ID}.asc'),outputControll)
    compareEyeTrackingWithBeh(EyetrackerDF,logs['story1_beh'],logs['story2_beh'],outputControll)
    outputControll.write("Testing EDF file Timing")
    outputControll.write("   Story1: ")
    testTiming(EyetrackerDF['story1'].to_numpy(),outputControll)
    outputControll.write("   Story2: ")
    testTiming(EyetrackerDF['story2'].to_numpy(),outputControll)

    el_tracker.close()  # Disconnect from EyeLink

pygame.quit()
exit()

 