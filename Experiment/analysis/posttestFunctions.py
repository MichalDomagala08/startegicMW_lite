import numpy as np
import os 
import pandas as pd  # Assuming pandas is used for DataFrame creation
from scipy.interpolate import CubicSpline
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
        own = [];
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

                    if len(currentLine)>2:
                        if 'KEYPRESS' in currentLine[2] or 'PART' in currentLine[2] or 'LISTEN' in currentLine[2] or 'RECALL' in currentLine[2]:
                            own.append(currentLine)
                    if 'GAZE_COORDS' in currentLine[1]:
                        gazeCoords = [float(msg.strip()) for i, msg in enumerate(currentLine[1].split(" ")) if i > 1]  # Extract gaze coordinates

    
        return own,rawEye, events, gazeCoords  # Return the extracted data

    files = os.listdir(path)  # List all files in the specified directory

    if name == "":
        # If no specific filename is provided, load all files in the directory
        gazeCoordsF = []
        rawEyeF = []
        eventsF = []
        ownF=[]
        for filename in files:
            ownTemp,rawEyeTemp, eventsTemp, gazeCoordsTemp = loadOrder(filename)
            gazeCoordsF.append(gazeCoordsTemp)
            rawEyeF.append(rawEyeTemp)
            eventsF.append(eventsTemp)
            ownF.append(ownTemp)
    else:
        # If a specific filename is provided, load only that file

        
        ownF,rawEyeF, eventsF, gazeCoordsF = loadOrder(name)

    return ownF,rawEyeF, eventsF, gazeCoordsF
path =r"C:\Users\barak\Documents\GitHub\startegicMW_lite\Experiment\data"

loadParticipant(path, name="melT1")


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
    chooseViz (str): String of bools indicating which visualizations to include (fixations, saccades, ).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(20, 10))
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
    
    # Calculate session duration in seconds
    sesDur = (ScreenData['TimePoint'].iloc[-1] - ScreenData['TimePoint'].iloc[0]) / 1000

    # Set x-axis ticks and labels
    original_ticks = np.linspace(ScreenData['TimePoint'].iloc[0], ScreenData['TimePoint'].iloc[-1], num=10)
    labels_in_seconds = (original_ticks - ScreenData['TimePoint'].iloc[0]) / 1000
    plt.xticks(
        ticks=original_ticks,
        labels=np.round(labels_in_seconds, 1)
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Pupil Diameter (a.u.)")
    plt.legend()

    return fig


    

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
    plt.show()

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



def removeBlinks(blinkDF, ScreenData, inter=50):
    """
    Remove blinks from data in a specified interval (default 50 ms).

    Parameters:
    blinkDF (DataFrame): DataFrame containing blink events.
    ScreenData (DataFrame): DataFrame containing screen data.
    inter (int): Interval in milliseconds to extend the blink period.

    Returns:
    DataFrame: ScreenData with blinks removed.
    """
    # Expand the blink interval by the specified interval before and after
    blinkDF["Beg_extended"] = blinkDF["Beg"] - inter
    blinkDF["End_extended"] = blinkDF["End"] + inter

    # Create a boolean array to mark blink intervals
    is_blink = np.zeros(len(ScreenData), dtype=bool)

    # Iterate over each blink interval and mark the corresponding time points
    for _, row in blinkDF.iterrows():
        is_blink |= (ScreenData["TimePoint"] >= row.Beg_extended) & (ScreenData["TimePoint"] <= row.End_extended)

    # Filter out the blink intervals
    ScreenData = ScreenData[~is_blink]

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

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt




def removeOutliers(ScreenData, threshold=3, max_duration=500, boundary=50, verbose=0,eye='LeftPupil',logfile=None):
    """
    Remove outliers in pupil size data using the 3-sigma rule on the differential time series.

    Parameters:
    ScreenData (DataFrame): DataFrame containing screen data.
    threshold (int): Threshold for identifying outliers (default 3 for 3-sigma rule).
    max_duration (int): Maximum duration for interpolation (default 500 ms).
    boundary (int): Interval in milliseconds to extend the outlier period.
    verbose (int): Verbosity level for debugging and visualization (default 0).

    Returns:
    DataFrame: ScreenData with interpolated outliers.
    """
    


    # Create differential time series for the left eye
    diff_left = np.diff(ScreenData[eye], prepend=ScreenData[eye].iloc[0])
    diff_right = np.diff(ScreenData[eye], prepend=ScreenData[eye].iloc[0])
    # Identify outliers using the 3-sigma rule on the differential time series
    mean_diff_left = np.mean(diff_left)
    std_diff_left = np.std(diff_left)
    outliers_left = abs(diff_left - mean_diff_left) > threshold * std_diff_left

    if logfile is not None:
        logfile.write(f"        (4) Interpolating or NaN Outliers in {eye} eye N:{len(outliers_left):5.1f}\n")
        print(f"    Interpolating or NaN Outliers in {eye} eye N:{len(outliers_left):5.1f}")
    if verbose ==2:
        plt.figure(figsize=(12, 6))
        plt.plot(diff_left, label='Left Pupil Size')
        plt.xlabel('TimePoint')
        plt.ylabel('Differential Time Series')
        plt.title('Left Pupil Size with Outliers')
        plt.legend()
        plt.show()

    # Visualize the outliers
    if verbose ==2:
        plt.figure(figsize=(12, 6))
        plt.plot(ScreenData['TimePoint'], ScreenData[eye], label='Left Pupil Size')
        plt.plot(ScreenData['TimePoint'], ScreenData['RightPupil'], label='Left Pupil Size')

        plt.scatter(ScreenData['TimePoint'][outliers_left], ScreenData[eye][outliers_left], color='red', label='Outliers')
        plt.xlabel('TimePoint')
        plt.ylabel('Left Pupil Size')
        plt.title('Left Pupil Size with Outliers')
        plt.legend()
        plt.show()

    # Cluster outliers together
    clusters = []
    current_cluster = []
    for i in range(len(outliers_left)):
        if outliers_left[i]:
            current_cluster.append(i)
        else:
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
    if current_cluster:
        clusters.append(current_cluster)

    # Create a mask for the outliers and their boundaries
    mask = np.zeros(len(ScreenData), dtype=bool)
    for clust in clusters:
        Beg_extended = ScreenData['TimePoint'].iloc[clust[0]] - boundary
        End_extended = ScreenData['TimePoint'].iloc[clust[-1]] + boundary
        mask |= (ScreenData["TimePoint"] >= Beg_extended) & (ScreenData["TimePoint"] <= End_extended)

    if verbose ==2:
        # Visualize the outliers with extension
        plt.figure(figsize=(12, 6))
        plt.plot(ScreenData.loc[mask, 'TimePoint'], ScreenData.loc[mask, eye], label='Left Pupil Size')
        plt.xlabel('TimePoint')
        plt.ylabel('Left Pupil Size')
        plt.title('Outliers only with extension')
        plt.legend()
        plt.show()

    # Set outlier values to NaNs
    ScreenData.loc[mask, eye] = np.nan

    # Interpolate using cubic splines
    valid_idx = ScreenData[eye].dropna().index  # Indices where data is NOT an artefact
    valid_x = valid_idx.to_numpy()
    valid_y = ScreenData.loc[valid_idx, eye].to_numpy()

    # Create cubic spline function
    cs = CubicSpline(valid_x, valid_y, extrapolate=False)  # Extrapolate False to account for missing data at the edges

    # Interpolate at artefactual indices
    ScreenData = ScreenData.copy()

    ScreenData.loc[mask, eye] = cs(ScreenData.index[mask])

    # Visualize the interpolated data
    if verbose ==2:
        plt.figure(figsize=(12, 6))
        plt.plot(ScreenData['TimePoint'], ScreenData[eye], label='Left Pupil Size')
        plt.scatter(ScreenData['TimePoint'][mask], ScreenData[eye][mask], color='red', label='Interpolated Points')
        plt.xlabel('TimePoint')
        plt.ylabel('Left Pupil Size')
        plt.title('Left Pupil Size with Interpolated Outliers')
        plt.legend()
        plt.show()

    # Iterate through interpolated clusters and check if any exceed max_duration
    for clust in clusters:
        Beg_extended = ScreenData['TimePoint'].iloc[clust[0]] - boundary
        End_extended = ScreenData['TimePoint'].iloc[clust[-1]] + boundary
        mask = (ScreenData["TimePoint"] >= Beg_extended) & (ScreenData["TimePoint"] <= End_extended)
        duration = ScreenData['TimePoint'][mask].iloc[-1] - ScreenData['TimePoint'][mask].iloc[0]
        if duration > max_duration:
            #print(f"Cluster from {Beg_extended} to {End_extended} exceeds max_duration with duration {duration} ms")
            ScreenData.loc[mask, eye] = np.nan

    # Visualize the final data
    if verbose ==2:
        plt.figure(figsize=(12, 6))
        plt.plot(ScreenData['TimePoint'], ScreenData[eye], label='Left Pupil Size')
        plt.xlabel('TimePoint')
        plt.ylabel('Left Pupil Size')
        plt.title('Left Pupil Size after Removing Long Clusters')
        plt.legend()
        plt.show()

    # Print the mask for debugging
   # print(mask)
    return ScreenData

from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def merge_blink_intervals(blinkDF, inter=50):
    """
    Merge overlapping or closely spaced blink events.

    Parameters:
    blinkDF (DataFrame): DataFrame containing blink events.
    inter (int): Interval in milliseconds to extend the blink period.

    Returns:
    DataFrame: Merged blink intervals.
    """
    # Sort by start time
    blinkDF = blinkDF.sort_values(by="Beg").copy()

    merged_intervals = []
    current_beg = blinkDF.iloc[0]["Beg"] - inter
    current_end = blinkDF.iloc[0]["End"] + inter

    for _, row in blinkDF.iterrows():
        new_beg = row["Beg"] - inter
        new_end = row["End"] + inter

        if new_beg <= current_end:  # Overlapping or close blink events
            current_end = max(current_end, new_end)
        else:
            merged_intervals.append([current_beg, current_end])
            current_beg = new_beg
            current_end = new_end

    # Append last merged interval
    merged_intervals.append([current_beg, current_end])

    # Convert back to DataFrame
    mergedDF = pd.DataFrame(merged_intervals, columns=["Beg_extended", "End_extended"])
    return mergedDF

def interpolate_blinks(blinkDF, ScreenData, inter=50,maxBlink=500,verbose = 0,logfile=None,begendDur = 1000,interp_method=0):
    if logfile is not None:
        logfile.write("        (3) Interpolating Blinks: \n")
        
    """
    Interpolate blinks from data in a specified interval (default 50 ms) using Cubic Spline interpolation.

    Parameters:
    blinkDF (DataFrame): DataFrame containing blink events.
    ScreenData (DataFrame): DataFrame containing screen data.
    inter (int): Interval in milliseconds to extend the blink period.

    Returns:
    DataFrame: ScreenData with interpolated blinks.
    """
    ScreenData = ScreenData.copy()
    removedBlinks = 0;
    interpBlinks=0;
    rempDur = 0;
    interpDur = 0;
    # Merge overlapping blink events before applying interpolation
    merged_blinks = merge_blink_intervals(blinkDF, inter)
    # Process each merged blink interval
    for i, row in merged_blinks.iterrows():
        beg_ext = row.Beg_extended
        end_ext = row.End_extended
        duration = end_ext - beg_ext
        mask = (ScreenData["TimePoint"] >= beg_ext) & (ScreenData["TimePoint"] <= end_ext)

        # Nans When Blink event is Too Long OR when it is at the Immediate Beginning!
        if  duration > maxBlink  and verbose == 2:
              print(f"Blink n. {i} is too long! Duration: {duration}")
              if logfile is not None:
                    logfile.write(f"            interpolating Blink: N: {i}  Duration: {duration}\n")

        if abs(ScreenData["TimePoint"].iloc[0] -beg_ext) < begendDur  and verbose == 2:
              print(f"  Blink {i} is too close to the beginning! ({ScreenData['TimePoint'].iloc[0] - beg_ext}")
              if logfile is not None:
                    logfile.write(f"            Blink {i} is too close to the beginning! ({ScreenData['TimePoint'].iloc[0] - beg_ext})\n")

        if abs(ScreenData["TimePoint"].iloc[-1] - end_ext) < begendDur and verbose == 2:
              print(f"Blink {i} is too close to the end! ({ScreenData['TimePoint'].iloc[-1] - end_ext})")
              if logfile is not None:
                    logfile.write(f"            Blink {i} is too close to the end! ({ScreenData['TimePoint'].iloc[-1] - end_ext})\n")


        if duration <= maxBlink and abs(ScreenData["TimePoint"].iloc[0] - beg_ext) > begendDur and abs(ScreenData["TimePoint"].iloc[-1] - end_ext) > begendDur:
            for col in ["LeftPupil", "RightPupil"]:
                valid_idx = ScreenData.loc[~mask, col].dropna().index
                if len(valid_idx) < 5:  # Need at least 5 valid points for cubic spline
                    ScreenData.loc[mask, col] = np.nan
               
                    continue
                if logfile is not None and verbose == 2:
                    logfile.write(f"            interpolating Blink: N: {i}  Duration: {duration}\n")
                valid_x = valid_idx.to_numpy()
                valid_y = ScreenData.loc[valid_idx, col].to_numpy()

                try:
                    if interp_method == 0:  # Linear interpolation
                        linear_interp = interp1d(valid_x, valid_y, bounds_error=False, fill_value="extrapolate")
                        ScreenData.loc[mask, col] = linear_interp(ScreenData.loc[mask].index)
                        interpBlinks +=1
                        interpDur += duration;

                    elif interp_method == 1:  # Cubic spline interpolation
                        cs = CubicSpline(valid_x, valid_y, extrapolate=False)
                        ScreenData.loc[mask, col] = cs(ScreenData.loc[mask].index)
                        interpDur += duration;
                        interpBlinks +=1
                except Exception as e:
                    if logfile is not None:
                        logfile.write(f"            Interpolation failed for {col} at {beg_ext}-{end_ext}: {e}\n")
                    print(f"Interpolation failed for {col} at {beg_ext}-{end_ext}: {e}")
                    ScreenData.loc[mask, col] = np.nan  # Fallback to NaN if error occurs
        else:
            removedBlinks +=1
            rempDur += duration;

            if logfile is not None  and verbose == 2:
                logfile.write(f"            Blink removed: N:{i}, Duration: {duration}\n")
            ScreenData.loc[mask, ["LeftPupil", "RightPupil"]] = np.nan

    if logfile is not None:
        logfile.write(f"            Blinks removed: {removedBlinks} (duration: {rempDur})\n")
        logfile.write(f"            Blinks interp:  {interpBlinks} (duration: {interpDur})\n")

    return ScreenData





def downsamplePupil(ScreenData,divisor=5,logfile=None):
    """
    Downsample Pupil Data to 100 Hz
    :param ScreenData:
    :return:
    """
    # Downsample to freq (default 100) Hz
    ScreenData = ScreenData[ScreenData['TimePoint'] % divisor == 0]
    if logfile is not None:
        print(f"    Downsampling to rate {500/divisor}")
        logfile.write(f"        (1) Downsampling to rate {500/divisor}\n")
    return ScreenData

def plotEyeWithBlink(RawDF, blinkDF, gazeCoords, titl="Raw Gaze Data", visual_angle=5):
    """
    Plot raw gaze data for the left eye, highlighting points during blinks and their 50ms boundary.
    Also, draw a circle to signify the 5 degrees of visual angle.

    Parameters:
    RawDF (DataFrame): Raw eye-tracking data.
    blinkDF (DataFrame): DataFrame containing blink events with 'Beg' and 'End' columns.
    gazeCoords (list): Screen dimensions [width, height, center_x, center_y].
    titl (str): Title of the plot.
    visual_angle (float): Visual angle threshold in degrees.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(20, 20))
    plt.title(titl, fontsize=20)

    plt.xlabel("Width Coords (pixels)")
    plt.ylabel("Height Coords (pixels)")

    # Create a mask for points during blinks and their 50ms boundary
    blink_mask = np.zeros(len(RawDF), dtype=bool)
    for _, blink in blinkDF.iterrows():
        start = blink["Beg"] - 50  # Extend 50ms before the blink
        end = blink["End"] + 50   # Extend 50ms after the blink
        blink_mask |= (RawDF["TimePoint"] >= start) & (RawDF["TimePoint"] <= end)

    # Plot left eye data
    plt.scatter(
        RawDF.loc[~blink_mask, 'LeftX'],  # Non-blink points
        RawDF.loc[~blink_mask, 'LeftY'],
        color='blue', label='Left Eye (Non-Blink)', s=10
    )
    plt.scatter(
        RawDF.loc[blink_mask, 'LeftX'],  # Blink points (including 50ms boundary)
        RawDF.loc[blink_mask, 'LeftY'],
        color='green', label='Left Eye (Blink + 50ms Boundary)', s=10
    )

    # Add a rectangle representing the screen dimensions
    rect = plt.Rectangle((0, 0), gazeCoords[2], gazeCoords[3], 
                         linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    # Calculate the radius of the circle for 5 degrees of visual angle
    center_x = gazeCoords[2] / 2
    center_y = gazeCoords[3] / 2
    radius_x = Dg2px(visual_angle, 70, 53, 1920)  # Horizontal radius in pixels
    radius_y = Dg2px(visual_angle, 70, 53, 1080)  # Vertical radius in pixels

    # Add a circle to signify the 5 degrees of visual angle
    circle = plt.Circle((center_x, center_y), radius_x, color='red', fill=False, linestyle='--', linewidth=2)
    plt.gca().add_patch(circle)

    plt.legend(loc='upper center')

    return fig


def replace_zero_clusters_with_nans(ScreenData, boundary=50,logfile=None):
    """
    Identify clusters of zeros in LeftPupil, replace them with NaN, and extend the replacement
    by a specified boundary (e.g., 50ms).

    Parameters:
    ScreenData (DataFrame): DataFrame containing gaze and pupil data.
    boundary (int): Time in milliseconds to extend the range around zero clusters.

    Returns:
    DataFrame: Modified ScreenData with zero clusters and their boundaries replaced by NaN.
    """
    import numpy as np

    # Identify rows where LeftPupil is zero
    zero_mask = (ScreenData['LeftPupil'] <= 0)
    if logfile is not None:
            logfile.write("        (5) Replacing Zeros: \n")
    # If no zeros are found, return the original DataFrame
    if not zero_mask.any():
        if logfile is not None:
            logfile.write("            No zeros present in the data\n")
        return ScreenData
    else:
        if logfile is not None:
            logfile.write(f"            Zeros: N:{zero_mask.sum():4.1f}\n")
    # Get the row positions (not the DataFrame index) of zeros
    zero_positions = np.where(zero_mask)[0]

    # Create a mask for the clusters and their boundaries
    boundary_mask = np.zeros(len(ScreenData), dtype=bool)

    # Iterate through zero positions and apply the boundary
    for pos in zero_positions:
        start_idx = max(0, pos - boundary)
        end_idx = min(len(ScreenData) - 1, pos + boundary)
        boundary_mask[start_idx:end_idx + 1] = True

    # Replace the clusters and their boundaries with NaN
    ScreenData.loc[boundary_mask, ['LeftPupil', 'RightPupil']] = np.nan

    return ScreenData

def smooth_pupil_data(ScreenData, window_size=5, min_cluster_duration=1000, max_gap_duration=3000,eye='LeftPupil',logfile=None,verbose=0):
    """
    Smooth the pupil data (LeftPupil and RightPupil) while ignoring NaNs.
    Each cluster of consecutive non-NaN measurements is smoothed separately.
    Clusters shorter than min_cluster_duration (in ms) or with gaps longer than max_gap_duration
    are replaced with NaN.

    Parameters:
    ScreenData (DataFrame): DataFrame containing gaze and pupil data.
    window_size (int): Size of the smoothing window (e.g., for moving average).
    min_cluster_duration (int): Minimum duration (in ms) for a cluster to be smoothed.
    max_gap_duration (int): Maximum gap (in ms) between timestamps within a cluster.

    Returns:
    DataFrame: Modified ScreenData with smoothed pupil data.
    """
    import numpy as np
    import pandas as pd

    if logfile is not None:
        logfile.write("        (6) Smoothin Out the Data: \n")
    def smooth_cluster(cluster, window_size):
        """Apply smoothing to a single cluster."""
        return cluster.rolling(window=window_size, center=True, min_periods=1).mean()

    # Copy the data to avoid modifying the original DataFrame
    smoothed_data = ScreenData.copy()

    # Identify clusters of consecutive non-NaN values
    non_nan_mask = pd.notna(ScreenData[eye])
    cluster_indices = np.where(non_nan_mask)[0]
    if len(cluster_indices) == 0:
        # If no valid data, skip smoothing for this column
        return smoothed_data

    # Group consecutive indices into clusters
    clusters = []
    current_cluster = []

    # Clustring
    for i in range(1,len(cluster_indices)):
        if cluster_indices[i] - cluster_indices[i-1] > 1:
            clusters.append(current_cluster)
            current_cluster = []
        else:
            current_cluster.append(cluster_indices[i] )

    # Smooth or replace each cluster
    if len(clusters) == 0 :
    
        if verbose > 0:
            print("    Warning -  there are no Clusters in data")
            if logfile is not None:
                logfile.write("            Warning -  there are no Clusters in data\n")
        return smoothed_data

    elif len(clusters[0]) == 0: 
        if verbose > 0:
            print("    Warning - Either there are no Clusters in data")
            if logfile is not None:
                logfile.write("            Warning -  the first cluster is 0 in length \n")
        return smoothed_data
    
    previousClusterEnd =  ScreenData.iloc[clusters[0]]['TimePoint'].iloc[-1]

    for cluster in clusters:
        # Calculate cluster duration
        cluster_timepoints = ScreenData.iloc[cluster]['TimePoint']
        if len(cluster_timepoints) == 0 :
        
            if verbose >0: # If there is no timepoints in a cluster
                print(f"    Warning - {cluster} cluster is empty")
                if logfile is not None:
                    logfile.write(f"            Warning - {cluster} cluster is empty\n")
    
            return smoothed_data

        cluster_duration = cluster_timepoints.iloc[-1] - cluster_timepoints.iloc[0]

        if cluster_duration < min_cluster_duration and cluster_timepoints.iloc[0] - previousClusterEnd > max_gap_duration:
            # If the cluster is too short, replace it with NaN
            smoothed_data.iloc[cluster, smoothed_data.columns.get_loc(eye)] = np.nan
        else:
            # Otherwise, smooth the cluster
            cluster_data = smoothed_data.iloc[cluster][eye]
            smoothed_data.iloc[cluster, smoothed_data.columns.get_loc(eye)] = smooth_cluster(cluster_data, window_size)
        previousClusterEnd = cluster_timepoints.iloc[-1]

    return smoothed_data


### Use this  to filter None Screen Data - at the end

def count_gaze_outside(ScreenData, blinkDF, gazeCoords, max_blink_duration=500, boundary=50, visual_angle=5,logfile=None,verbose=0):
    """
    Count the number of gaze points outside 5 degrees of visual angle
    that do not coincide with a blink (plus 50ms boundary), are not longer than 500ms,
    and are not NaN.

    Parameters:
    ScreenData (DataFrame): DataFrame containing gaze and pupil data.
    blinkDF (DataFrame): DataFrame containing blink events.
    gazeCoords (list): List containing screen dimensions [width, height, center_x, center_y].
    max_blink_duration (int): Maximum blink duration (in ms) to exclude from counting.
    boundary (int): Time in milliseconds to extend the range around blink periods.
    visual_angle (float): Visual angle threshold in degrees.

    Returns:
    int: Number of gaze points outside the visual angle that meet the criteria.
    """


    import numpy as np
    ScreenData = ScreenData.copy()

    # Calculate the center of the screen
    center_x = gazeCoords[2] / 2
    center_y = gazeCoords[3] / 2

    # Calculate the Degrees Angle of the LeftX and LeftY values
    std_x = Dg2px(visual_angle, 70, 53, 1920)
    std_y = Dg2px(visual_angle, 70, 53, 1920)

    # Identify points outside the visual angle
    outside_mask = ~(
        (np.abs(ScreenData['LeftX'] - center_x) < std_x) &
        (np.abs(ScreenData['LeftY'] - center_y) < std_y)
    )
    # Exclude NaN values from the mask
    not_nan_mask = pd.notna(ScreenData['LeftX']) & pd.notna(ScreenData['LeftY'])

    # Combine the outside mask with the not-NaN mask
    outside_mask = outside_mask & not_nan_mask

    # Create a mask for blink periods (including 50ms boundary)
    blink_mask = np.zeros(len(ScreenData), dtype=bool)

    for i, blink in blinkDF.iterrows():
        if blink['Duration'] <= max_blink_duration:
            start = blink['Beg'] - boundary
            end = blink['End'] + boundary
            blink_mask |= (ScreenData["TimePoint"] >= start) & (ScreenData["TimePoint"] <= end)

    if logfile is not None and verbose > 0:
        logfile.write(f"        (2) Removing Points from Outside of the Gaze by {visual_angle} dg ({std_x:3.1f} px) \n")
        logfile.write(f"            N: {blink_mask.sum():5.1f};  \n")

    if verbose > 0:
        print(f"    Removing Points from Outside of the Gaze by {visual_angle} dg. ")
    # Exclude blink periods from the outside mask
    valid_outside_mask = outside_mask & ~blink_mask
    # Count the number of valid gaze points outside the visual angle
    count_outside = valid_outside_mask.sum()
    ScreenData.loc[valid_outside_mask,'LeftPupil'] = np.nan
    return ScreenData




def Dg2px(dg,dist,width,wpx):
    return wpx*2*dist*np.tan(dg*np.pi/360)/width

def px2Dg(px,dist,width,wpx):
    return 360*np.arctan(px*width/(2*dist*wpx))/np.pi




def changeTiming(RawDF,fixationDF,saccadesDF,blinkDF,own):
    # Get the time of the event

    fixationDF['Beg'] = fixationDF['Beg'] -RawDF['TimePoint'].iloc[0]
    fixationDF['End'] = fixationDF['End'] -RawDF['TimePoint'].iloc[0]

    saccadesDF['Beg'] = saccadesDF['Beg'] -RawDF['TimePoint'].iloc[0]
    saccadesDF['End'] = saccadesDF['End'] -RawDF['TimePoint'].iloc[0]

    blinkDF['Beg'] = blinkDF['Beg'] -RawDF['TimePoint'].iloc[0]
    blinkDF['End'] = blinkDF['End'] -RawDF['TimePoint'].iloc[0]
    own = [[o[0], (int(o[1].strip()) - RawDF['TimePoint'].iloc[0])] + o[2:] for o in own]

    RawDF['TimePoint'] = RawDF['TimePoint'] - RawDF['TimePoint'].iloc[0]

    return RawDF,fixationDF,saccadesDF,blinkDF,own




def parse_events(own):
    """
    Parse the events from the own list and separate them into different DataFrames.

    Parameters:
    own (list): List of lists from the ASC file corresponding to custom events.

    Returns:
    dict: Dictionary containing DataFrames for each event type.
    """
    parts = [];
    end = [];
    beg = [];
    keypress = [];
    keytime = [];
    story = [];

    # Parse the own list
    for line in own:
        event_type = line[2]
        timestamp = int(line[1])
        event_name = line[3].strip()
        event_status = line[4].strip()
        
        #print(line)
        if event_type == 'PART'and event_name not in parts:
            parts.append(event_name)
            beg.append(timestamp)
            story.append(latestStory)
        if event_type =='PART'and event_status == 'END':
            end.append(timestamp)
        if event_type == 'KEYPRESS':
            if event_name == '49':
                keypress.append("FOCUS")
            elif event_name == '50':
                keypress.append("TRT")
            elif event_name == '51':
                keypress.append("MW")
            elif event_name == '52':
                keypress.append("MB")
            keytime.append(timestamp)
        if event_type =='LISTEN' and event_name not in story:
            latestStory = event_name
   
    # Create DataFrames for each event type
    event_dfs = pd.DataFrame(zip(*[parts,beg, end,keypress, keytime,story]),columns =['Part','beg','end','key','keytime','story'] )
    return event_dfs




def split_dataframes_by_events(RawDF, fixationDF, saccadesDF, blinkDF, events_df):
    """
    Split the RawDF, fixationDF, saccadesDF, and blinkDF into separate DataFrames based on the time intervals in events_df.

    Parameters:
    RawDF (DataFrame): DataFrame containing raw eye-tracking data.
    fixationDF (DataFrame): DataFrame containing fixation events.
    saccadesDF (DataFrame): DataFrame containing saccade events.
    blinkDF (DataFrame): DataFrame containing blink events.
    events_df (DataFrame): DataFrame containing event intervals with BEG and END columns.

    Returns:
    dict: Dictionary containing split DataFrames for each event interval.
    """
    split_data = {
       'STORY_1' : {'RawDF': {}, 'fixationDF': {}, 'saccadesDF': {}, 'blinkDF': {}},
       'STORY_2' : {'RawDF': {}, 'fixationDF': {}, 'saccadesDF': {}, 'blinkDF': {}}
    }

    for idx, row in events_df.iterrows():
        beg = row['beg']
        end = row['end']
        event_name = row['Part']
        story_name = row['story']
        split_data[story_name]['RawDF'][event_name] = RawDF[(RawDF['TimePoint'] >= beg) & (RawDF['TimePoint'] <= end)]
        split_data[story_name]['fixationDF'][event_name] = fixationDF[(fixationDF['Beg'] >= beg) & (fixationDF['End'] <= end)]
        split_data[story_name]['saccadesDF'][event_name] = saccadesDF[(saccadesDF['Beg'] >= beg) & (saccadesDF['End'] <= end)]
        split_data[story_name]['blinkDF'][event_name] = blinkDF[(blinkDF['Beg'] >= beg) & (blinkDF['End'] <= end)]

    return split_data




def preprocessingPipeline(blinkDF,RawDF,saccadesDF,gazeCoords,story,part,fixationDF=None,log_file=[],pdfs=[None],verbose=0,interp_type=0,interpBoundary=50,maxBlinkDur=500,resampleRate=100,dgvCenter=5,smoothwin=5,min_cluster_duration=2000,max_gap_duration=500):


    # 1) Downsampling Raw Data to limit computation time
    RawDFDown = downsamplePupil(RawDF,logfile=log_file,divisor=500/resampleRate)
    if verbose :
        fig1 = plotPupilTimecourse(RawDFDown, f"Downsampled Pupil Data: {story} // {part}", blinkDF, saccadesDF, fixationDF, chooseViz='011')
        if len(pdfs) >= 1:
            pdfs[0].savefig(fig1)
        else:
            plt.show(fig1)
        plt.close(fig1)

    # 2) Compute Non-blink Outside of Center points
    ScreenData = count_gaze_outside(RawDFDown, blinkDF, gazeCoords, max_blink_duration=maxBlinkDur, boundary=interpBoundary, visual_angle=dgvCenter,logfile=log_file,verbose=verbose)
    if verbose:
        fig2 = plotEyeWithBlink(ScreenData[pd.notna(ScreenData['LeftPupil'])], blinkDF, gazeCoords, f"Screen Data: {story} // {part}")
        if len(pdfs) >= 2:
            pdfs[1].savefig(fig2)
        else:
            plt.show(fig2)
        plt.close(fig2)

    # 3) Interpolate Blinks
    RawDF2 = interpolate_blinks(blinkDF=blinkDF, ScreenData=ScreenData, inter=interpBoundary,maxBlink = 2*interpBoundary+maxBlinkDur,verbose =verbose,logfile=log_file,interp_method=interp_type).copy()
    if verbose:
        fig3 = plotPupilTimecourse(RawDF2, f"Interpolated Blinks: {story} // {part}", blinkDF, saccadesDF, fixationDF, chooseViz='011')
        if len(pdfs) >= 3:
            pdfs[2].savefig(fig3)
        else:
            plt.show(fig3)
        plt.close(fig3)

    # 4) Remove Other Outliers (it doesnt Work irregardless :C)
    RawDF3 = removeOutliers(RawDF2, threshold=3, max_duration=maxBlinkDur, boundary=40, verbose=verbose,logfile=log_file)
    if verbose:
        fig4 = plotPupilTimecourse(RawDF3, f"Outliers Removed: {story} // {part}", blinkDF, saccadesDF, fixationDF, chooseViz='011')
        if  len(pdfs) >= 4:
            pdfs[3].savefig(fig4)
        else:
            plt.show(fig4)
        plt.close(fig4)

    # 5) Remove Edge NaNs + boundary
    RawDF4 = replace_zero_clusters_with_nans(RawDF3,logfile=log_file)
    if verbose:
        fig5 = plotPupilTimecourse(RawDF4, f"Edge Artifacts Removed: {story} // {part}", blinkDF, saccadesDF, fixationDF, chooseViz='011')
        if len(pdfs) >= 5:
            pdfs[4].savefig(fig5)
        else:
            plt.show(fig5)
        plt.close(fig5)

    # 6) Smooth the data
    finalData = smooth_pupil_data(RawDF4, window_size=smoothwin, min_cluster_duration=min_cluster_duration, max_gap_duration=max_gap_duration, eye='LeftPupil',logfile=log_file,verbose=verbose)
    if verbose:
        fig6 = plotPupilTimecourse(finalData, f"Smoothed Data: {story} // {part}", blinkDF, saccadesDF, fixationDF, chooseViz='011')
        if len(pdfs) >= 6:
            pdfs[5].savefig(fig6)
        else:
            plt.show(fig6)

        plt.close(fig6)
    
    return finalData

