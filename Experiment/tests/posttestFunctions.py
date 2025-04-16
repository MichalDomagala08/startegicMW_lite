import numpy as np
import os 
import pandas as pd  # Assuming pandas is used for DataFrame creation
import matplotlib.pyplot as plt

        ########################################################
        ##### ------------  LOADING FUNCTIONs ------------ #####
        ########################################################


def loadParticipant(path, name=""):
    """
    Load participant data from the specified directory.

    Parameters:
    path (str): The directory path where participant files are located.
    name (str, optional): The specific filename to load. If not provided, all files in the directory will be loaded.

    Returns:
    tuple: A tuple containing raw eye data, events, and gaze coordinates.
    """
    def loadOrder(filename):
        """
        Load data from a specific file and extract raw eye data, events, and gaze coordinates.

        Parameters:
        filename (str): The name of the file to load.
        """
        rawEye = []
        events = []
        gazeCoords = []

        with open(os.path.join(path, filename, filename  +'.asc'), 'r') as currFile:
            allLines = currFile.readlines()  # Read all lines from the file
            for line in allLines:
                currentLine = line.split("\t")  # Split each line by tab delimiter

                ### Get Raw Eye Data
                if len([i for i in currentLine[0] if i.isdigit()]) == len(currentLine[0]) and len(currentLine[0]) != 0:
                    currentCell = []
                    for i in range(len(currentLine) - 1):
                        if currentLine[i].strip() != '.':
                            currentCell.append(float(currentLine[i].strip()))  # Convert and append numeric values
                        else:
                            currentCell.append(np.nan)  # Append NaN for missing values
                    rawEye.append(currentCell)
                
                ### Get Events
                elif len([i for i in currentLine[0] if i.isdigit()]) != 0:
                    if 'ESACC' in currentLine[0] or 'EFIX' in currentLine[0] or 'EBLINK' in currentLine[0]:
                        events.append(currentLine)  # Append event data

                # Get Screen Parameters
                if currentLine[0] == 'MSG':
                    if 'GAZE_COORDS' in currentLine[1]:
                        gazeCoords = [float(msg.strip()) for i, msg in enumerate(currentLine[1].split(" ")) if i > 1]  # Extract gaze coordinates

    
        return rawEye, events, gazeCoords  # Return the extracted data

    files = os.listdir(path)  # List all files in the specified directory

    if name == "":
        # If no specific filename is provided, load all files in the directory
        gazeCoordsF = []
        rawEyeF = []
        eventsF = []
        for filename in files:
            rawEyeTemp, eventsTemp, gazeCoordsTemp = loadOrder(filename)
            gazeCoordsF.append(gazeCoordsTemp)
            rawEyeF.append(rawEyeTemp)
            eventsF.append(eventsTemp)
    else:
        # If a specific filename is provided, load only that file

        
        rawEyeF, eventsF, gazeCoordsF = loadOrder(name)

    return rawEyeF, eventsF, gazeCoordsF

def getEvent(events):
    """
    Process event data to extract saccades, fixations, and blinks.

    Parameters:
    events (list): A list of event data.

    Returns:
    tuple: A tuple containing DataFrames for fixations, saccades, and blinks.
    """
    saccades = []  # Initialize an empty list to store saccades
    fixations = []  # Initialize an empty list to store fixations
    blink = []  # Initialize an empty list to store blinks
    
    for j, ev in enumerate(events):
        if 'ESACC' in ev[0]:
            # Process saccade events
            saccades.append([ev[0][6], float(ev[0][9:])] + [float(ev[i].strip()) for i in range(1, len(ev))])
        elif 'EFIX' in ev[0]:
            # Process fixation events
            if j > 1:
                fixations.append([ev[0][5], float(ev[0][9:])] + [float(ev[i].strip()) for i in range(1, len(ev))])
        elif 'EBLINK' in ev[0]:
            # Process blink events
            blink.append([ev[0][7], float(ev[0][9:])] + [float(ev[i].strip()) for i in range(1, len(ev))])

    # Data Frame For Fixation Events
    fixationDF = pd.DataFrame(fixations, columns=['Eye', 'Beg', 'End', 'Length', 'X', 'Y', 'Disp'])

    # Data Frame For Saccade Events
    saccadesDF = pd.DataFrame(saccades, columns=['Eye', 'Beg', 'End', 'Duration', 'StartX', 'StartY', 'EndX', 'EndY', 'Amplitude', 'Velocity'])

    # Data Frame For Blink Events
    blinkDF = pd.DataFrame(blink, columns=['Eye', 'Beg', 'End', 'Duration'])

    return fixationDF, saccadesDF, blinkDF  # Return the DataFrames


 


        ##############################################################
        ##### ------------  VIZUALISATION FUNCTIONs ------------ #####
        ##############################################################
    

def plotFixData(fixPosL, fixPosR, gazeCoords,titl="Raw Fixation Data"):
    """
    Plot fixation data for left and right eyes.

    Parameters:
    fixPosL (DataFrame): Fixation data for the left eye.
    fixPosR (DataFrame): Fixation data for the right eye.
    gazeCoords (list): List containing gaze coordinates.
    """
    plt.figure(figsize=(20, 20 * 1080 / 1920))
    plt.title(titl, fontsize=20)

    plt.xlabel("Width Coords (pixels)")
    plt.ylabel("Height Coords (pixels)")
    
    # Plot fixation points for left and right eyes
    plt.plot(fixPosL['X'], fixPosL['Y'], 'b.', label='Left')
    plt.plot(fixPosR['X'], fixPosR['Y'], 'r.', label='Right')
    
    plt.legend(loc='upper center')
    
    # Add a rectangle representing the screen dimensions
    rect = plt.Rectangle((0, 0), gazeCoords[2], gazeCoords[3], 
                         linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)
    


import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def eyeRawKDE(RawDF,titl="Raw Gaze Distributions"):
    """
    Create kernel density plots for raw eye-tracking data.

    Parameters:
    RawDF (DataFrame): Raw eye-tracking data.
    """
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(16, 10)
    fig.suptitle(titl,fontsize=20)

    # KDE plot for X coordinates
    sb.kdeplot(RawDF['LeftX'], ax=axs[0])
    sb.kdeplot(RawDF['RightX'], ax=axs[0])
    axs[0].set_title('X')
    axs[0].legend(loc='upper left', labels=['Left', 'Right'])

    # KDE plot for Y coordinates
    sb.kdeplot(RawDF['LeftY'], ax=axs[1])
    sb.kdeplot(RawDF['RightY'], ax=axs[1])
    axs[1].set_title('Y')
    axs[1].legend(loc='upper left', labels=['Left', 'Right'])

    # KDE plot for pupil diameter
    sb.kdeplot(RawDF['LeftPupil'], ax=axs[2])
    sb.kdeplot(RawDF['RightPupil'], ax=axs[2])
    axs[2].set_title('Pupil')
    axs[2].legend(loc='upper left', labels=['Left', 'Right'])



def plotRawGaze(RawDF,gazeCoords,titl="Raw Gaze Data"):
    """
    Plot raw gaze data for left and right eyes.

    Parameters:
    RawDF (DataFrame): Raw eye-tracking data.
    """
    plt.figure(figsize=(20, 20))
    plt.title(titl, fontsize=20)

    plt.xlabel("Width Coords (pixels)")
    plt.ylabel("Height Coords (pixels)")
    
    # Plot raw gaze data for left and right eyes
    plt.plot(RawDF['LeftX'], RawDF['LeftY'], label='Left')
    plt.plot(RawDF['RightX'], RawDF['RightY'], 'r--', label='Right')
    
    plt.legend(loc='upper center')
    
    # Add a rectangle representing the screen dimensions
    rect = plt.Rectangle((0, 0), gazeCoords[2], gazeCoords[3], 
                         linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)
    

import matplotlib.pyplot as plt
import numpy as np

def plotPupilTimecourse(ScreenData, title, blinks, saccades, fixations, chooseViz='011'):
    """
    Plot pupil diameter over time with optional visualization of fixations, blinks, and saccades.

    Parameters:
    ScreenData (DataFrame): Screen data containing time points and pupil diameter.
    title (str): Title of the plot.
    blinks (DataFrame): DataFrame containing blink events.
    saccades (DataFrame): DataFrame containing saccade events.
    fixations (DataFrame): DataFrame containing fixation events.
    chooseViz (str): String of bools indicating which visualizations to include (fixations, blinks, saccades).
    """
    plt.figure(figsize=(20, 10))
    plt.title(title, fontsize=20)
    
    # Plot pupil diameter data
    plt.plot(ScreenData['TimePoint'], ScreenData['LeftPupil'], label="Left Pupil", color='blue')
    
    # Add fixation rectangles
    if bool(int(chooseViz[0])):
        for _, fixation in fixations.iterrows():
            plt.axvspan(fixation['Beg'], fixation['End'], color='lightblue', alpha=0.2, label="Fixation" if _ == 0 else "")
    
    # Add saccade rectangles
    if bool(int(chooseViz[1])):
        for _, saccade in saccades.iterrows():
            plt.axvspan(saccade['Beg'], saccade['End'], color='green', alpha=0.3, label="Saccade" if _ == 0 else "")
    
    # Add blink rectangles
    if bool(int(chooseViz[2])):
        for _, blink in blinks.iterrows():
            plt.axvspan(blink['Beg'], blink['End'], color='red', alpha=0.3, label="Blink" if _ == 0 else "")
    
    plt.xlabel("Time-Points (ms)")
    plt.ylabel("Pupil Diameter (a.u.)")
    plt.legend()


    

def plotSaccadeData(left_saccades, right_saccades, gazeCoords):
    """
    Plot saccade data for left and right eyes.

    Parameters:
    left_saccades (DataFrame): DataFrame containing saccade data for the left eye.
    right_saccades (DataFrame): DataFrame containing saccade data for the right eye.
    gazeCoords (list): List containing gaze coordinates [width, height].
    """
    # Create a figure with specified size
    plt.figure(figsize=(20, 20 * 1080 / 1920))
    plt.xlabel("Width Coords (pixels)")
    plt.ylabel("Height Coords (pixels)")

    # Plot left saccades (blue)
    for _, row in left_saccades.iterrows():
        plt.arrow(row["StartX"], row["StartY"], 
                  row["EndX"] - row["StartX"], row["EndY"] - row["StartY"], 
                  head_width=10, head_length=15, color='blue', alpha=0.7)
    
    # Plot right saccades (red)
    for _, row in right_saccades.iterrows():
        plt.arrow(row["StartX"], row["StartY"], 
                  row["EndX"] - row["StartX"], row["EndY"] - row["StartY"], 
                  head_width=10, head_length=15, color='red', alpha=0.7)
    
    # Add legend
    plt.scatter([], [], color='blue', label="Left Eye Saccades")
    plt.scatter([], [], color='red', label="Right Eye Saccades")
    plt.legend(loc='upper center')

    # Add rectangle for screen boundary
    rect = plt.Rectangle((0, 0), gazeCoords[2], gazeCoords[3], 
                         linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)
    
    # Display the plot

        #########################################################
        ##### ------------  Cleaning Functions ------------ #####
        #########################################################
    

def matchLeftRight(DF):
    """
        function to match Timings of Left and Right Events in our DataFrame
    """
    # Convert DataFrame to NumPy arrays
    left = DF[DF["Eye"] == "L"].to_numpy()
    right = DF[DF["Eye"] == "R"].to_numpy()

    # Extract relevant columns (timestamps)
    left_beg = left[:, 1]  # Beginning timestamps for left eye
    left_end = left[:, 2]  # End timestamps for left eye

    right_beg = right[:, 1]  # Beginning timestamps for right eye
    right_end = right[:, 2]  # End timestamps for right eye

    # Define time threshold for matching
    time_threshold = 40

    # Create a pairwise difference matrix between left and right fixations (!!!) 
    beg_diff_matrix = np.abs(left_beg[:, None] - right_beg)  # Broadcasted subtraction
    end_diff_matrix = np.abs(left_end[:, None] - right_end)  # Broadcasted subtraction

    # Find valid matches where both beg and end differences are within threshold
    valid_matches = (beg_diff_matrix <= time_threshold) & (end_diff_matrix <= time_threshold)

    # Get indices of matching pairs
    left_indices, right_indices = np.where(valid_matches)

    # Store matched fixations
    matched_fixations_np = np.hstack((left[left_indices], right[right_indices]))

    # Create DataFrame for easier readability
    matched_fixations_df = pd.DataFrame(
        matched_fixations_np,
        columns=["L_" + i for i in  DF.columns.to_list() ] +
                ["R_" + i for i in  DF.columns.to_list() ]
    )

   
    fixPosL = pd.DataFrame(left[left_indices],columns=DF.columns.to_list())
    fixPosR = pd.DataFrame( right[right_indices],columns=DF.columns.to_list())
    # Ensure correct data types
    for col in DF.columns:
        if col != "Eye":
            fixPosL[col] = fixPosL[col].astype(float)
            fixPosR[col] = fixPosR[col].astype(float)


    return fixPosL, fixPosR, matched_fixations_df



def removeBlinks(blinkDF,ScreenData,inter = 50):

    """
        Remove Blinks from Data in a specified Interval (default 50 ms)

    """
    # Expand the blink interval by 20 ms before and after
    blinkDF["Beg_extended"] = blinkDF["Beg"] - inter
    blinkDF["End_extended"] = blinkDF["End"] + inter

    # Function to check if a timepoint falls within any blink interval
    def is_within_blink(timepoint, blink_intervals):
        return any((timepoint >= row.Beg_extended) and (timepoint <= row.End_extended) for _, row in blink_intervals.iterrows())

    # Apply filtering
    ScreenData["is_blink"] = ScreenData["TimePoint"].apply(lambda tp: is_within_blink(tp, blinkDF))
    ScreenData = ScreenData[~ScreenData["is_blink"]].drop(columns=["is_blink"])  # Keep only valid rows

    return ScreenData

def cleanStd(ScreenData,obj,stdThesh):
    """
        Gets data that Exceeds a certain threshold of the standard deviation
        Requires a DataFrame containing Data, and a list of columns to check (Always 2) as well as a Threshold
    """
    artefactualStdPupil = ScreenData[list(abs(ScreenData[obj[0]].values - np.mean(ScreenData[obj[0]].values))> stdThesh*np.std(ScreenData[obj[0]].values)) 
           and list(abs(ScreenData[obj[1]].values - np.mean(ScreenData[obj[1]].values))> stdThesh*np.std(ScreenData[obj[1]].values))]
    return artefactualStdPupil


def cleanIQR(ScreenData,obj,threshold = 1.5,windows =0 ):
    """
        Computes IQR on my samples - DOESNT WORK WELL WITH THIS KIND OF DATA....
        Requires a DataFrame of values to filter through, a threshold and a window size 
        as well as OBJ whiich is a name of the column on basis which the filter is used
    """
    if windows == 0:
        windows = len(ScreenData[obj])
    IQRartifs = [];

    for i in range(0,len(ScreenData[obj]),windows):
        window = ScreenData[obj].iloc[i:i+windows];
        Q1 = window.quantile(0.25)
        Q3 = window.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        for j in range(len(window)):
            if window.iloc[j] < lower_bound or window.iloc[j]> upper_bound:
                IQRartifs.append(i+j)

    ScreenDataCleaned = ScreenData.drop(np.intersect1d(set(ScreenData.index.to_numpy()),set(IQRartifs)),axis=0)
    return IQRartifs,ScreenDataCleaned


        ######################################################
        ##### ------------ Checks Functions ------------ #####
        ######################################################

def checkSaccadeMismatch(saccades):
    diffX = []
    diffY = []
    for i in range(1, len(saccades)):
        # Check X
        diffX.append(abs(saccades.iloc[i]['StartX'] - saccades.iloc[i-1]['EndX']))
        # Check Y
        diffY.append(abs(saccades.iloc[i]['StartY'] - saccades.iloc[i-1]['EndY']))

    saccades['diffX'] = [0] + diffX
    saccades['diffY'] = [0] + diffY

    return saccades

