import numpy as np
import os 
import pandas as pd  # Assuming pandas is used for DataFrame creation
from scipy.interpolate import CubicSpline,interp1d
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
        if 'ESACC' in ev[0] and ev[3].strip() != '.':
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




def changeTiming(RawDF,fixationDF,saccadesDF,blinkDF,own):
    """
        OLD FUNCTION
    """
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
        #print(event_type,event_name)
        #print(line)
        if event_type == 'PART' and event_status != 'END':
            parts.append(event_name)
            beg.append(timestamp)
            story.append(latestStory)
        if event_type =='PART'and event_status == 'END':
            end.append(timestamp)
        if event_type == 'KEYPRESS':
            keypress.append(int(event_name))
            keytime.append(timestamp)
        if event_type =='LISTEN' and event_name not in story:
            latestStory = event_name
   
    # Create DataFrames for each event type
    event_dfs = pd.DataFrame(zip(*[parts,beg, end,keypress, keytime,story]),columns =['Part','beg','end','key','keytime','story'] )

    return event_dfs

def rename_parts(df):

    """
    rename parts in the DataFrame by removing the number at the end and renumbering them.
    (A sanity check, when the parts have been incorrectly numbered)
    :param df: DataFrame containing the parts to be renamed.
    """
    df['name'] = df['Part'].str.rsplit('_', n=1).str[0]
    # Add renumbering per name - just count the number of occurences of each entity and cumulate it counting! 
    df['new_num'] = df.groupby('name').cumcount() + 1
    # Rebuild 'parts' column
    df['Part'] = df['name'] + '_' + df['new_num'].astype(str)
    # Drop helper columns if you want
    df = df.drop(columns=['name', 'new_num'])
    return df

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
 


def splitEntities(RawDF, event_dfs):
    """
    Splits the RawDF into two entities based on the events DataFrame.Dependning who the story has been about
    """
    firstEntityRaw  = []
    secondEntityRaw = []

    for i in range(len(event_dfs)):

        currentTiming1 = event_dfs['beg'].iloc[i]
        currentTiming2 = event_dfs['end'].iloc[i]

        if 'TimePoint' in RawDF.columns:
            RawDF1 = RawDF[(RawDF['TimePoint'] >= currentTiming1) & (RawDF['TimePoint'] <= currentTiming2)]
        elif 'Beg' in RawDF.columns:
            RawDF1 = RawDF[(RawDF['Beg'] >= currentTiming1) & (RawDF['End'] <= currentTiming2)]

        if i %2 ==0:
            firstEntityRaw.append(RawDF1)
        else:
            secondEntityRaw.append(RawDF1)

    firstEntityDF = pd.concat(firstEntityRaw, ignore_index=True)
    secondEntityDF = pd.concat(secondEntityRaw, ignore_index=True)

    return firstEntityDF, secondEntityDF



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

    return fig

def plotRawGaze(RawDF,gazeCoords,titl="Raw Gaze Data"):
    """
    Plot raw gaze data for left and right eyes.

    Parameters:
    RawDF (DataFrame): Raw eye-tracking data.
    """
    fig = plt.figure(figsize=(20, 20))
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
    return fig

import matplotlib.pyplot as plt
import numpy as np
def plotPupilTimecourse(ScreenData, title, blinks, saccades, fixations, chooseViz='011',secChan=0):
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
    if secChan:
        plt.plot(ScreenData['TimePoint'], ScreenData['RightPupil'], label="Right Pupil", color='orange')

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



def plotEyeWithBlink(RawDF, blinkDF, gazeCoords, titl="Raw Gaze Data", visual_angle=5):
    """
    Plot raw gaze data for the left eye, highlighting points during blinks and their 50ms boundary.
    Also, draw a circle to signify the 5 degrees of visual angle.

    Parameters:
    RawDF        (DataFrame): Raw eye-tracking data.
    blinkDF      (DataFrame): DataFrame containing blink events with 'Beg' and 'End' columns.
    gazeCoords   (list):      Screen dimensions [width, height, center_x, center_y].
    titl         (str):       Title of the plot.
    visual_angle (float):    Visual angle threshold in degrees.
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




def plot_gaze_check(P_dict,filename,tracked):
    """
    Plots the gaze check for each entity in the story. in 4 different plots:
    Differentiates between Tracked and Untracked Entity 
    """
    # Set up subplots: 4 rows, 1 column
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    plt.suptitle(f"Gaze Checks: {filename}", fontsize=16)
    if tracked == 'KAROLINA':
        label1 = 'Tracked'
        label2 = 'Untracked'
    else:
        label2 = 'Tracked'
        label1 = 'Untracked'
    # Loop through each key and its corresponding data
    for ax, (key, P) in zip(axs, P_dict.items()):
        x = list(range(len(P)))
        x_even = x[::2]
        y_even = P[::2]
        x_odd = x[1::2]
        y_odd = P[1::2]

        ax.scatter(x_even, y_even, color='blue', label=label1)
        ax.scatter(x_odd, y_odd, color='orange', label=label2)
        ax.set_title(f"{key}", fontsize=14)
        ax.set_ylabel("Value")
        ax.grid(True)

    # Add shared X label
    axs[-1].set_xlabel("Trial Number")
    axs[0].legend(loc='upper right')




def mwHist(resultsDF,filename):
    """
    Plots a general histogram to see whether there is a disproporitons of particular answers
    """
    plt.figure(figsize=(12,12))
    plt.title(f'Tracked/Untracked Count: {filename}')
    plt.hist(resultsDF[resultsDF['Tracking'] == 'TRACKED']['MW Estimate'], bins=20,
            label='Tracked', alpha=0.5,color='blue')  # Set alpha to 0.5 for transparency
    plt.hist(resultsDF[resultsDF['Tracking'] == 'UNTRACKED']['MW Estimate'], bins=20, 
            label='Untracked', alpha=0.5,color='orange') # Set alpha to 0.5 for transparency
    plt.legend()
    plt.xlabel('MW Estimate')
    plt.ylabel('Count')

def scatterResults(MW,Results, filename, pdf=None):
    """
    Plots pupil diameter for tracked and untracked entities in a 2x2 grid.
    
    Parameters:
    - Results: List of results to plot (e.g., [resultsL, resultsR, resultsL2, resultsR2]).
    - filename: Name of the subject entity.
    - pdf: PdfPages object to save the plots (optional).
    """
    fig, axs = plt.subplots(int( len(Results)/2),2, figsize=(12, 12),sharey='row')
    plt.suptitle(f"Pupil Diameter for {filename}", fontsize=16)

    # Titles for subplots
    row_titles = ["Mean", "Std",'diff']
    col_titles = ["Left Eye", "Right Eye"]

    for row in range(2):  # Rows: Mean vs. Std
        for col in range(2):  # Columns: Left Eye vs. Right Eye
            ax = axs[row, col]
            result = Results[row * 2 + col]  # Select the correct result (e.g., resultsL, resultsR, resultsL2, resultsR2)

            # Scatter plot for tracked and untracked entities
            ax.scatter(MW[::2], result[::2], label="Tracked Entity", color="blue")
            ax.scatter(MW[1::2], result[1::2], label="Untracked Entity", color="orange")

            # Set titles, labels, and grid
            if row == 0:
                ax.set_title(col_titles[col], fontsize=14)
            if col == 0:
                ax.set_ylabel(f"{row_titles[row]} Pupil Diameter (a.u.)", fontsize=12)
            ax.set_xlabel("MW Estimate (1-100)", fontsize=12)
            ax.legend()
            ax.grid(True)

    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the title

    if pdf is not None:
        pdf.savefig(fig)


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

    OLD  and UNOPTIMISED
    
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
        OLD AND UNOPTIMISED

        Gets data that Exceeds a certain threshold of the standard deviation
        Requires a DataFrame containing Data, and a list of columns to check (Always 2) as well as a Threshold
    """
    artefactualStdPupil = ScreenData[list(abs(ScreenData[obj[0]].values - np.mean(ScreenData[obj[0]].values))> stdThesh*np.std(ScreenData[obj[0]].values)) 
           and list(abs(ScreenData[obj[1]].values - np.mean(ScreenData[obj[1]].values))> stdThesh*np.std(ScreenData[obj[1]].values))]
    return artefactualStdPupil


def cleanIQR(ScreenData,obj,threshold = 1.5,windows =0 ):
    """
        OLD AND UNOPTIMISED

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



def GeneralChecks(RawDF, blinkDF, saccadesDF, fixationDF, gazeCoords,log_file=None):
    """
    General Checks for the Raw Data
    """

    # Check if the data is in the right format
    if not isinstance(RawDF, pd.DataFrame):
        raise ValueError("RawDF should be a pandas DataFrame")
    if not isinstance(blinkDF, pd.DataFrame):
        raise ValueError("blinkDF should be a pandas DataFrame")
    if not isinstance(saccadesDF, pd.DataFrame):
        raise ValueError("saccadesDF should be a pandas DataFrame")
    if not isinstance(fixationDF, pd.DataFrame):
        raise ValueError("fixationDF should be a pandas DataFrame")
    

    from analysisFunctions import eyeRawKDE,plotRawGaze
    import matplotlib.pyplot as plt 

    RawDFPupilTrue = RawDF[(RawDF['LeftPupil'] > 0) & (RawDF['RightPupil'] > 0)]


    #. Check whether Majority of Gaze Positions Lay in the center
    if log_file is not None:
        print("\n(1) Check mean Gaze Center\n")
        log_file.write("\n(1) Check mean Gaze Center\n")

        print("                              X          Y")
        log_file.write("                              X           Y\n")
        print(f"Middle Screen Coordinates: {(gazeCoords[2] / 2):6.1f}      {gazeCoords[3] / 2:6.1f}")
        log_file.write(f"Middle Screen Coordinates: {(gazeCoords[2] / 2):6.1f}      {gazeCoords[3] / 2:6.1f}\n")
        print(f"Mean Eye Coordinates:      {np.mean(RawDF['LeftX']):5.1f}       {np.mean(RawDF['LeftY']):5.1f}")
        log_file.write(f"Mean Eye Coordinates:      {np.mean(RawDF['LeftX']):5.1f}       {np.mean(RawDF['LeftY']):5.1f}\n")

        # 1. Check if there are Too Many Missing data from One Eye
        print("\n(2) Check Missing Pupil Data as a proxy for missing!\n")
        log_file.write("\n(2) Check Missing Pupil Data as a proxy for missing!\n")
        zeroPupilLeft = RawDF[RawDF['LeftPupil'] == 0]
        zeroPupilRight = RawDF[RawDF['RightPupil'] == 0]
        zeroPupilBoth = RawDF[(RawDF['RightPupil'] == 0) & (RawDF['LeftPupil'] == 0)]
        print(f"Number of NoGaze Data: \n(Left: {len(zeroPupilLeft)}, Right: {len(zeroPupilRight)}), Both: {len(zeroPupilBoth)}")
        log_file.write(f"Number of NoGaze Data: \n(Left: {len(zeroPupilLeft)}, Right: {len(zeroPupilRight)}), Both: {len(zeroPupilBoth)}\n")

        log_file.write("\n"+"-"*100 + "\n")
    

        # (1) Mismatch between Pupil Diameters Left to Right 
        # Get raw when all data are greater than 0 (no blinks) 
        pupilDifference = (RawDFPupilTrue['LeftPupil'] - RawDFPupilTrue['RightPupil']) / RawDFPupilTrue['LeftPupil']
        print("\n (3) Pupil Difference Test \n")
        log_file.write("\n (3) Pupil Difference Test \n")
        print(f"    Percentage Difference: {np.mean(pupilDifference)}")
        log_file.write(f"    Percentage Difference: {np.mean(pupilDifference)}\n")


    return RawDFPupilTrue,log_file


#[RawDFPupilTrue,log_file] = GeneralChecks(RawDF, blinkDF, saccadesDF, fixationDF, gazeCoords)


def qualiryMetrics(RawDFs, saccadesDFs, blinkDFs, gazeCoords, log_file=None,verbose=1):
    nogazeRate = np.empty([2, 2])
    gazeDiff = np.empty([2, 2])
    pupilDiff = np.empty([2, 2])
    saccDiff = np.empty([2, 2])
    blinkDiff = np.empty([2, 2])
    eyeposDiff = np.empty([2, 2])

    for i in range(len(RawDFs)):
        df = RawDFs[i]

        for j, LR in enumerate(['Left', 'Right']):
            # Gaze center distance
            dx = gazeCoords[2] / 2 - np.mean(df[LR + 'X'])
            dy = gazeCoords[3] / 2 - np.mean(df[LR + 'Y'])
            gazeDiff[i, j] = np.sqrt(dx**2 + dy**2)

            # No gaze data ratio
            zeroPupil = df[df[LR + 'Pupil'] == 0]
            nogazeRate[i, j] = len(zeroPupil) / len(df)

            # Saccade and blink counts
            saccDiff[i, j] = np.sum(saccadesDFs[i]['Eye'] == LR[0])
            blinkDiff[i, j] = np.sum(blinkDFs[i]['Eye'] == LR[0])

        # Pupil difference
        pupilValid = df[(df['LeftPupil'] > 0) & (df['RightPupil'] > 0)]
        pupilDiff[i, 0] = len(pupilValid['LeftPupil'])
        pupilDiff[i, 1] = len(pupilValid['RightPupil'])
        pupil_rel_diff = (pupilValid['LeftPupil'] - pupilValid['RightPupil']) / pupilValid['LeftPupil']
        pupil_mean = np.mean(pupil_rel_diff)
        pupil_std = np.std(pupil_rel_diff)

        # Eye position mismatch
        eye_diff = abs(df['LeftX'] - df['RightX'])
        mean_eye_diff = np.mean(eye_diff)
        std_eye_diff = np.std(eye_diff)
        eyeposDiff[i, 0] = len(df['LeftX'])
        eyeposDiff[i, 1] = len(df['RightX'])

        if verbose and log_file is not None: 

            # === LOG & PRINT ===
            print(f"\n    ({i+1}) Quality Metrics:\n")
            log_file.write(f"\n    ({i+1}) Quality Metrics:\n")

            # Define fixed width formatting
            def fmt(val): return f"{val:>10.5f}"
            def fmt_int(val): return f"{val:>10}"
            print(f"{'    Metric':<30} {'Left':>10} {'Right':>10}")
            log_file.write(f"{'    Metric':<30} {'Left':>10} {'Right':>10}\n")

            print(f"{'    Coordinate Difference:':<30} {fmt(gazeDiff[i, 0])} {fmt(gazeDiff[i, 1])}")
            log_file.write(f"{'    Coordinate Difference:':<30} {fmt(gazeDiff[i, 0])} {fmt(gazeDiff[i, 1])}\n")

            print(f"{'    Rate of No Gaze Data:':<30} {fmt(nogazeRate[i, 0])} {fmt(nogazeRate[i, 1])}")
            log_file.write(f"{'    Rate of No Gaze Data:':<30} {fmt(nogazeRate[i, 0])} {fmt(nogazeRate[i, 1])}\n")

            print(f"{'    Saccade Count:':<30} {fmt_int(int(saccDiff[i, 0]))} {fmt_int(int(saccDiff[i, 1]))}")
            log_file.write(f"{'    Saccade Count:':<30} {fmt_int(int(saccDiff[i, 0]))} {fmt_int(int(saccDiff[i, 1]))}\n")

            print(f"{'    Blink Count:':<30} {fmt_int(int(blinkDiff[i, 0]))} {fmt_int(int(blinkDiff[i, 1]))}\n")
            log_file.write(f"{'    Blink Count:':<30} {fmt_int(int(blinkDiff[i, 0]))} {fmt_int(int(blinkDiff[i, 1]))}\n\n")

            print(f"{'    Pupil Relative Diff (L-R):':<30} {pupil_mean:>10.5f} (±{pupil_std:.5f})")
            log_file.write(f"{'    Pupil Relative Diff (L-R):':<30} {pupil_mean:>10.5f} (±{pupil_std:.5f})\n")

            print(f"{'    Eye Position Mismatch (X):':<30} {mean_eye_diff:>10.5f} (±{std_eye_diff:.5f})\n")
            log_file.write(f"{'    Eye Position Mismatch (X):':<30} {mean_eye_diff:>10.5f} (±{std_eye_diff:.5f})\n")

    return gazeDiff, pupilDiff, saccDiff, blinkDiff, eyeposDiff, nogazeRate,pupil_mean,mean_eye_diff




#--------------------------------------------------------------------------------------------------------------------------------------------------------------- # 


                                        ####################################################################
                                        ##### ------------  PUPIL PREPROCESSINF FUNCTIONS ------------ #####
                                        ####################################################################



def preprocessingPipeline(blinkDF,RawDF,saccadesDF,gazeCoords,story="",part="",fixationDF=None,log_file=[],pdfs=[None],verbose=0,interp_type=0,interpBoundary=50,maxBlinkDur=500,resampleRate=100,dgvCenter=5,smoothwin=5,min_cluster_duration=2000,max_gap_duration=50):
    """
    Preprocessing pipeline for eye-tracking data.

    This function processes raw eye-tracking and event data to produce a cleaned, interpolated, and smoothed pupil timecourse for both eyes. 
    It handles downsampling, removal of out-of-bounds gaze points, blink interpolation, outlier removal, edge artifact cleaning, and smoothing. 
    Optionally, it generates diagnostic plots at each step.

    Inputs:
        blinkDF         : DataFrame containing blink events for our trial
        RawDF           : DataFrame containing raw eye-tracking data for our trial
        saccadesDF      : DataFrame containing saccade events  for our trial
        gazeCoords      : List or array with gaze/screen coordinates. [0 0 width height]
     

        #General Parameters#
        fixationDF      : (Optional) DataFrame containing fixation events. IIt is usefull only for plotting! 
                                     Default = None
        log_file        : (Optional) File handle or list for logging. If you want to Log the process
                                     Default = None
        pdfs            : (Optional) List of PdfPages or None for saving plots.
                                     Default = []
        verbose         : (Optional) Verbosity level for debug/plots. If you want plots set it to 1
                                     Default = 1
        story           : (Optional) String identifier for the story/condition. (Used Only as a placeholder for PLOTS)
        part            : (Optional) String or int identifier for the part/trial. (Used Only as a placeholder for PLOTS)

        #Downsampling#
        resampleRate        : (Optional) Resample rate for downsampling (Hz).
                                         Default = 100
        #Blink Interpolation#
        interp_type     : (Optional) Interpolation type for blinks (0=linear, 1=cubic). 
                                     Default (and recommended) =  0 (linear)
        interpBoundary  : (Optional) Boundary (ms) around blinks for interpolation.
                                     Default = 50ms 
        maxBlinkDur     : (Optional) Maximum blink duration for interpolation (ms).
                                      If it is exceeded, events are NaN-ed.
                                      Default = 500ms
        #Centering Gaze#
        dgvCenter           : (Optional) Visual angle threshold for gaze exclusion (deg) . 
                                         How far from center is considered good Pupil diameter). 
                                         Default = 5
        #Smoothing#
        smoothwin           : (Optional) Window size for smoothing. How many samples are with smoothing over. 
                                          Default  = 5 (recommend 10)
        min_cluster_duration: (Optional) Minimum cluster duration for smoothing (ms). If it is exceeded, data is NaN-ed. 
                                          Default  = 2s
        max_gap_duration    :  (Optional) Maximum gap duration for smoothing (ms). 
                                          Concatenate data divided by NaNs for smoothing that are closer than max_gap. 
                                          Default = 50ms
    Returns:
        finalData       : DataFrame with cleaned and processed pupil data.
    """
    # 1) Downsampling Raw Data to limit computation time (WARNING! It assumes that your Fs is 500!)
    RawDFDown = downsamplePupil(RawDF,logfile=log_file,divisor=500/resampleRate)
    if verbose : # Plot if Necessery
        fig1 = plotPupilTimecourse(RawDFDown, f"Downsampled Pupil Data: {story} // {part}", blinkDF, saccadesDF, fixationDF, chooseViz='011',secChan=1)
        if len(pdfs) >= 1:
            pdfs[0].savefig(fig1)
        else:
            plt.show(fig1)
        plt.close(fig1)

    # 2) Remove Data that are not in the center (Do not count blinks yet!)
    ScreenData = count_gaze_outside(RawDFDown, blinkDF, gazeCoords, max_blink_duration=maxBlinkDur, boundary=interpBoundary, visual_angle=dgvCenter,logfile=log_file,verbose=verbose)
    if verbose:
        fig2 = plotEyeWithBlink(ScreenData[pd.notna(ScreenData['LeftPupil'])], blinkDF, gazeCoords, f"Screen Data: {story} // {part}")
        if len(pdfs) >= 2:
            pdfs[1].savefig(fig2)
        else:
            plt.show(fig2)
        plt.close(fig2)

    # 3) Interpolate Blinks
    RawDF2,interpDuration = interpolate_blinks(blinkDF=blinkDF, ScreenData=ScreenData, inter=interpBoundary,maxBlink = 2*interpBoundary+maxBlinkDur,verbose =verbose,logfile=log_file,interp_method=interp_type)
    if verbose:
        fig3 = plotPupilTimecourse(RawDF2, f"Interpolated Blinks: {story} // {part} ({interpDuration})", blinkDF, saccadesDF, fixationDF, chooseViz='011',secChan=1)
        if len(pdfs) >= 3:
            pdfs[2].savefig(fig3)
        else:
            plt.show(fig3)
        plt.close(fig3)

    # 4) Remove Other Outliers (it doesnt Work irregardless :C)
    RawDF3 = removeOutliers(RawDF2, threshold=3, max_duration=maxBlinkDur, boundary=40, verbose=verbose,logfile=log_file)
    if verbose:
        fig4 = plotPupilTimecourse(RawDF3, f"Outliers Removed: {story} // {part}", blinkDF, saccadesDF, fixationDF, chooseViz='011',secChan=1)
        if  len(pdfs) >= 4:
            pdfs[3].savefig(fig4)
        else:
            plt.show(fig4)
        plt.close(fig4)

    # 5) Remove Edge NaNs (Which are already padded!)
    RawDF4 = replace_zero_clusters_with_nans(RawDF3,logfile=log_file)
    if verbose:
        fig5 = plotPupilTimecourse(RawDF4, f"Edge Artifacts Removed: {story} // {part}", blinkDF, saccadesDF, fixationDF, chooseViz='011',secChan=1)
        if len(pdfs) >= 5:
            pdfs[4].savefig(fig5)
        else:
            plt.show(fig5)
        plt.close(fig5)


    # 6) Smooth the data
    finalData = smooth_pupil_data(RawDF4, window_size=smoothwin, min_cluster_duration=min_cluster_duration, max_gap_duration=max_gap_duration,logfile=log_file,verbose=verbose)
    if verbose:
        fig6 = plotPupilTimecourse(finalData, f"Smoothed Data: {story} // {part}", blinkDF, saccadesDF, fixationDF, chooseViz='011',secChan=1)
        if len(pdfs) >= 6:
            pdfs[5].savefig(fig6)
        else:
            plt.show(fig6)

        plt.close(fig6)
    
    return finalData



def downsamplePupil(ScreenData,divisor=5,logfile=None):
    """
    Downsample Pupil Data to 100 Hz
    :param ScreenData:
    : divisor - dividing current Rate to a given count via Modullo 
    : logfile (Optional) - if you want to log things 
    :return:
    """
    # Downsample to freq (default 100) Hz
    ScreenData = ScreenData[ScreenData['TimePoint'] % divisor == 0]
    if logfile is not None:
        print(f"    Downsampling to rate {500/divisor} Hz")
        logfile.write(f"        (1) Downsampling to rate {500/divisor} Hz\n")
    return ScreenData



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
    OR Linear interpolation

    First we MERGE left and RIGHT blinks: They are mostly corresponding with oneathoer

    Parameters:
    blinkDF (DataFrame): DataFrame containing blink events.
    ScreenData (DataFrame): DataFrame containing screen data.
    inter (int): Interval in milliseconds to extend the blink period.

    Returns:
    DataFrame: ScreenData with interpolated blinks.

    """
    interpDur = 0;

    if len(blinkDF) > 0:
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
    else:
         if logfile is not None:
            logfile.write("          No blinks to interpolate!!!!")
        
    return ScreenData,interpDur


### Use this  to filter None Screen Data - at the end

def count_gaze_outside(ScreenData, blinkDF, gazeCoords,max_blink_duration=500, boundary=50, visual_angle=5,logfile=None,verbose=0):
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
    std_x = Dg2px(visual_angle, 93, 53, 1920)
    std_y = Dg2px(visual_angle, 93, 53, 1920)


    for eye  in ['Left','Right']:
        # Identify points outside the visual angle
        outside_mask = ~(
            (np.abs(ScreenData[eye +'X'] - center_x) < std_x) &
            (np.abs(ScreenData[eye +'Y'] - center_y) < std_y)
        )
        # Exclude NaN values from the mask
        not_nan_mask = pd.notna(ScreenData[eye +'X']) & pd.notna(ScreenData[eye +'Y'])

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
        ScreenData.loc[valid_outside_mask,eye +'Pupil'] = np.nan
    return ScreenData




def Dg2px(dg,dist,width,wpx):
    return wpx*2*dist*np.tan(dg*np.pi/360)/width

def px2Dg(px,dist,width,wpx):
    return 360*np.arctan(px*width/(2*dist*wpx))/np.pi


def replace_zero_clusters_with_nans(ScreenData,logfile=None):
    """
    Identify clusters of zeros in LeftPupil and rightPupil Respectively, replace them with NaN, and extend the replacement
    by a specified boundary (e.g., 50ms).

    Parameters:
    ScreenData (DataFrame): DataFrame containing gaze and pupil data.
    boundary (int): Time in milliseconds to extend the range around zero clusters.

    Returns:
    DataFrame: Modified ScreenData with zero clusters and their boundaries replaced by NaN.
    """
    import numpy as np

    # Identify rows where LeftPupil is zero
    for eye in ['LeftPupil','RightPupil']:
        zero_mask = (ScreenData[eye] <= 0)
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

        # Iterate through zero positions
        for pos in zero_positions:
            start_idx = max(0, pos)
            end_idx = min(len(ScreenData) - 1, pos)
            boundary_mask[start_idx:end_idx + 1] = True

        # Replace the clusters and their boundaries with NaN
        ScreenData.loc[boundary_mask, [eye]] = np.nan

    return ScreenData


def smooth_pupil_data(ScreenData, window_size=5, min_cluster_duration=1000, max_gap_duration=50, logfile=None, verbose=0):
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
        logfile.write("        (6) Smoothing Out the Data: \n")

    def smooth_cluster(cluster, window_size):
        """Apply smoothing to a single cluster."""
        return cluster.rolling(window=window_size, center=True, min_periods=1).mean()

    # Copy the data to avoid modifying the original DataFrame
    smoothed_data = ScreenData.copy()

    for eye in ['LeftPupil', 'RightPupil']:
        non_nan_mask = pd.notna(ScreenData[eye])
        cluster_indices = np.where(non_nan_mask)[0]

        if len(cluster_indices) == 0:
            if logfile is not None:
                logfile.write(f"       WARNING! No valid data found for {eye}.\n")
            continue

        # Group consecutive indices into clusters
        clusters = []
        current_cluster = [cluster_indices[0]]
        for i in range(1, len(cluster_indices)):
           
            t_now  = ScreenData.iloc[cluster_indices[i]]['TimePoint']
            t_prev = ScreenData.iloc[cluster_indices[i-1]]['TimePoint']
            if t_now - t_prev > max_gap_duration:  # Gap detected
                clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append(cluster_indices[i])

        if current_cluster:
            clusters.append(current_cluster)

        if verbose > 0:
            print(f"    Found {len(clusters)} clusters for {eye}.")
            if logfile is not None:
                logfile.write(f"        Found {len(clusters)} clusters for {eye}.\n")

        previous_cluster_end = None

        for cluster in clusters:
            cluster_timepoints = ScreenData.iloc[cluster]['TimePoint']
            cluster_duration = cluster_timepoints.iloc[-1] - cluster_timepoints.iloc[0] 

            if previous_cluster_end is not None:
                gap_duration = cluster_timepoints.iloc[0] - previous_cluster_end
            else:
                gap_duration = None
            # Adding +1 sample to avoid rejectin by the nick of hair
            if cluster_duration + int( ScreenData.iloc[1]['TimePoint'] - ScreenData.iloc[0]['TimePoint'] ) < min_cluster_duration:
                # Replace cluster with NaN if it's too short or the gap is too large
                smoothed_data.iloc[cluster, smoothed_data.columns.get_loc(eye)] = np.nan
                if verbose > 0:
                    print(f"    Cluster replaced with NaN for {eye}: Duration={cluster_duration}, Gap={gap_duration}.")
                    if logfile is not None:
                        logfile.write(f"        Cluster replaced with NaN for {eye}: Duration={cluster_duration}, Gap={gap_duration}.\n")
            else:
                # Smooth the cluster
                cluster_data = smoothed_data.iloc[cluster][eye]
                smoothed_data.iloc[cluster, smoothed_data.columns.get_loc(eye)] = smooth_cluster(cluster_data, window_size)
                if verbose > 0:
                    print(f"    Cluster smoothed for {eye}: Duration={cluster_duration}.")
                    if logfile is not None:
                        logfile.write(f"        Cluster smoothed for {eye}: Duration={cluster_duration}.\n")

            previous_cluster_end = cluster_timepoints.iloc[-1]

    return smoothed_data



def removeOutliers(ScreenData, threshold=3, max_duration=500, boundary=50, verbose=0,logfile=None):
    """
    Remove outliers in pupil size data using the 3-sigma rule on the differential time series. \
        Replaces them with NaNs or interpolating them, if their duration has not corssed max_duration

    Parameters:
    ScreenData (DataFrame): DataFrame containing screen data.
    threshold (int): Threshold for identifying outliers (default 3 for 3-sigma rule).
    max_duration (int): Maximum duration for interpolation (default 500 ms).
    boundary (int): Interval in milliseconds to extend the outlier period.
    verbose (int): Verbosity level for debugging and visualization (default 0).

    Returns:
    DataFrame: ScreenData with interpolated outliers.
    """
    

    for eye in ['LeftPupil', 'RightPupil']:
        # Create differential time series for the left eye
        diff_left = np.diff(ScreenData[eye], prepend=ScreenData[eye].iloc[0])
        # Identify outliers using the 3-sigma rule on the differential time series
        mean_diff_left = np.mean(diff_left)
        std_diff_left = np.std(diff_left)
        outliers_left = abs(diff_left - mean_diff_left) > threshold * std_diff_left

        if logfile is not None:
            logfile.write(f"        (4) Interpolating or NaN Outliers in {eye} eye N:{len(outliers_left):5.1f}\n")
            print(f"    Interpolating or NaN Outliers in {eye} eye N:{len(outliers_left):5.1f}")
        if verbose ==2:
            plt.figure(figsize=(12, 6))
            plt.plot(diff_left, label=' Pupil Size')
            plt.xlabel('TimePoint')
            plt.ylabel('Differential Time Series')
            plt.title(eye+' Pupil Size with Outliers')
            plt.legend()
            plt.show()

        # Visualize the outliers
        if verbose ==2:
            plt.figure(figsize=(12, 6))
            plt.plot(ScreenData['TimePoint'], ScreenData[eye], label=eye+' Pupil Size')
            plt.plot(ScreenData['TimePoint'], ScreenData['RightPupil'], label=eye+' Pupil Size')

            plt.scatter(ScreenData['TimePoint'][outliers_left], ScreenData[eye][outliers_left], color='red', label='Outliers')
            plt.xlabel('TimePoint')
            plt.ylabel(eye+' Pupil Size')
            plt.title(eye+' Pupil Size with Outliers')
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
            plt.plot(ScreenData.loc[mask, 'TimePoint'], ScreenData.loc[mask, eye], label=eye+' Pupil Size')
            plt.xlabel('TimePoint')
            plt.ylabel(eye+' Pupil Size')
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
            plt.plot(ScreenData['TimePoint'], ScreenData[eye], label=eye+' Pupil Size')
            plt.scatter(ScreenData['TimePoint'][mask], ScreenData[eye][mask], color='red', label='Interpolated Points')
            plt.xlabel('TimePoint')
            plt.ylabel(eye+' Pupil Size')
            plt.title(eye+' Pupil Size with Interpolated Outliers')
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
            plt.plot(ScreenData['TimePoint'], ScreenData[eye], label=eye+' Pupil Size')
            plt.xlabel('TimePoint')
            plt.ylabel(eye+' Pupil Size')
            plt.title(eye+' Pupil Size after Removing Long Clusters')
            plt.legend()
            plt.show()

        # Print the mask for debugging
    # print(mask)
    return ScreenData

#--------------------------------------------------------------------------------------------------------------------------------------------------------------- # 
        
        
        #########################################################
        ##### ------------  Analysis Functions ------------ #####
        #########################################################




def wilcTest(x,y,verbose=0,log_file=None,analysisName=''):
    """
    Perform Wilcoxon signed-rank test on two paired samples.

    """
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(x, y)

    diff = y - x

    ### Compute Zvalue 
    # Remove zero differences (as scipy does)
    non_zero_diff = diff[diff != 0]

    # n = number of non-zero diffs
    n = len(non_zero_diff)

    # Expected value and standard deviation under null
    mean_T = n * (n + 1) / 4
    std_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    # Compute z-score
    z = (stat - mean_T) / std_T
    if verbose:
        print(f"---- Signed-rank test results  for {analysisName}: ----")
        print(f"Wilcoxon signed-rank test statistic: {stat}")
        print(f"p-value: {p}")
        print(f"Z-score: {z}\n")

    if log_file is not None:
        log_file.write(f"---- Signed-rank test results  for {analysisName}: ----")
        log_file.write(f"Wilcoxon signed-rank test statistic: {stat}")
        log_file.write(f"p-value: {p}")
        log_file.write(f"Z-score: {z}\n")


    return stat, p,z


def wilcLoop(resultsDF,entityTracked,verbose=0,log_file=None):
    """
        makes a Wilcoxon test of Tracked vs Untracked data on 2 analysis scheme:
        1) Whether they differ in MW subejctive Estimation
        2) Whether they differ on Pupil Diameter Metrics 

        Script for usage on a singular Person: Within Subject Analysis!
    """
    p =  []
    z = []
    s = []
    mean1 =[]
    mean2 =[]
    sd1 =[]
    sd2 = []
    anName = []
    parts = []
    modes = []
    eyes = []

    for part in ['first','second','All']:
        if part == 'first':
            cutt = slice(0, len(resultsDF) //4)  
        elif part == 'second':
            cutt =  slice(len(resultsDF) // 4, len(resultsDF)//2)
        elif part == 'All':
            cutt = slice(0, len(resultsDF)//2) 

        resPMW_W = wilcTest(resultsDF[resultsDF['Tracking'] == 'TRACKED']['MW Estimate'].iloc[cutt],
                            resultsDF[resultsDF['Tracking'] == 'UNTRACKED']['MW Estimate'].iloc[cutt],
                            verbose,analysisName='MW Estimate')
        p.append(resPMW_W[1])
        mean1.append(resultsDF[resultsDF['Tracking'] == 'TRACKED']['MW Estimate'].iloc[cutt].mean())
        mean2.append(resultsDF[resultsDF['Tracking'] == 'UNTRACKED']['MW Estimate'].iloc[cutt].mean())
        sd1.append(resultsDF[resultsDF['Tracking'] == 'TRACKED']['MW Estimate'].iloc[cutt].std())
        sd2.append(resultsDF[resultsDF['Tracking'] == 'UNTRACKED']['MW Estimate'].iloc[cutt].std())

        z.append(resPMW_W[2])
        s.append(resPMW_W[0])
        anName.append('MW Estimate')
        parts.append(part)
        modes.append('')
        eyes.append('')
        for j,mode in enumerate(['Mean','Std','diff']):
        
            

            for i,eye in enumerate(['Left Eye','Right Eye']):
                print(f"\nTests for Eye: {eye} and Estim Type {mode}")
                print("==="*20)
                if entityTracked == 'Karolina': # She is the First Entity
                    resPD_W  = wilcTest(resultsDF[resultsDF['Tracking'] == 'TRACKED']['Pupil Diameter'][mode][eye].iloc[cutt],
                                        resultsDF[resultsDF['Tracking'] == 'UNTRACKED']['Pupil Diameter'][mode][eye].iloc[cutt],
                                        verbose,analysisName='Pupil Diameter',log_file=log_file)
                

                else:
                    resPD_W  = wilcTest(resultsDF[resultsDF['Tracking'] == 'TRACKED']['Pupil Diameter'][mode][eye].iloc[cutt],
                                        resultsDF[resultsDF['Tracking'] == 'UNTRACKED']['Pupil Diameter'][mode][eye].iloc[cutt],
                                        verbose,analysisName='Pupil Diameter',log_file=log_file)
                p.append(resPD_W[1])
                z.append(resPD_W[2])
                s.append(resPD_W[0])
                mean1.append(resultsDF[resultsDF['Tracking'] == 'TRACKED']['Pupil Diameter'][mode][eye].iloc[cutt].mean())
                mean2.append(resultsDF[resultsDF['Tracking'] == 'UNTRACKED']['Pupil Diameter'][mode][eye].iloc[cutt].mean())
                sd1.append(resultsDF[resultsDF['Tracking'] == 'TRACKED']['Pupil Diameter'][mode][eye].iloc[cutt].std())
                sd2.append(resultsDF[resultsDF['Tracking'] == 'UNTRACKED']['Pupil Diameter'][mode][eye].iloc[cutt].std())

                anName.append('Pupil Diameter')
                parts.append(part)
                modes.append(mode)
                eyes.append(eye)

        stats_df = pd.DataFrame({
            'Statistic': s,
            'p-value': p,
            'Z-score': z
        })

    # Create a MultiIndex DataFrame
    index = pd.MultiIndex.from_arrays([ anName, parts, modes, eyes], names=['Analysis Name', 'Part', 'Mode', 'Eye'])
    stats_df = pd.DataFrame({'Statistic': s, 'p-value': p, 'Z-score': z ,'meanT':mean1,'meanUT':mean2,'meanT':sd1,'meanUT':sd2}, index=index)

    # Display the DataFrame
    pd.set_option('display.max_rows', 50)

    stats_df.sort_values(by=['Analysis Name'], inplace=True)
    return stats_df


##### --- Pupil Diameter Timecourse in the last part of the Experiment --- #####


from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def meanPupilShade(last10sDF1,last10sDF2,ds='',pvals = [], mltpl = 0,nsamp = 1000,sf=100):
    """
        Creates a Shade for the Mean Pupil Diameter in the last 10 seconds of the Experiment. For both Eyes

        Inputs:
    
        last10sDF1 - list of Last 10 seconds of the pupil diameter data in condition 1 across trials 
        last10sDF2 - Last 10 seconds of the pupil diameter in condition 2
        ds - Dataset Name (For vis)

    """

    # Compute mean of the last 10s for both conditions
    mean_df1 = (
        pd.concat(last10sDF1, keys=range(len(last10sDF1)))
        .groupby(level=1)
        .mean()
    )
    mean_df2 = (
        pd.concat(last10sDF2, keys=range(len(last10sDF2)))
        .groupby(level=1)
        .mean()
    )


    fig,axs = plt.subplots(1,2,figsize=(14,7))
    plt.suptitle(f"Mean Pupil Diameter in Tracked and Untracked Condition {ds}",fontsize=18)

    for i,eye in enumerate(['LeftPupil','RightPupil']):
        
        last10arr1 = np.array([last10sDF1[a][eye] for a in range(len(last10sDF1))]) # Make an array for more efficient computation 
        last10arr2 = np.array([last10sDF2[a][eye] for a in range(len(last10sDF2))])

        ### Wilcoxon Tests: (Making WIlcoxon for EVERY POINT)
        if len(pvals) == 0: # If pvals is empty, we can compute the stats
            stats = []
            for j in range(nsamp+1): # Iterate thtough timepoints
                stat, pv= wilcoxon(last10arr1[:,j],last10arr2[:,j], zero_method="wilcox",nan_policy='omit')
                stats.append(stat)
                pvals.append(pv)

        if mltpl:
            reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh") #Benjamini-Hochberg procedure for multiple comparison 
        else:
            p_fdr = pvals

            
        axs[i].plot(np.linspace(1,nsamp/sf,nsamp+1),mean_df1[eye],label='',color='blue')
        axs[i].fill_between(np.linspace(1,nsamp/sf,nsamp+1), 
                        mean_df1[eye] - np.nanstd(last10arr1[:][:],axis=0), 
                        mean_df1[eye]+ np.nanstd(last10arr1[:][:],axis=0), 
                        color='darkblue', alpha=0.2, label='TRACKED (STD Envelope)') # make an STD Shade 

        axs[i].plot(np.linspace(1,nsamp/sf,nsamp+1),mean_df2[eye],color='orange')
        axs[i].fill_between(np.linspace(1,nsamp/sf,nsamp+1), 
                        mean_df2[eye] - np.nanstd(last10arr2[:][:],axis=0), 
                        mean_df2[eye]+ np.nanstd(last10arr2[:][:],axis=0), 
                        color='orangered', alpha=0.2, label='TRACKED (STD Envelope)') # makeand STD shade 
        axs[i].set_title(eye)
        mask = np.array(p_fdr) < 0.05
        mask_nan = np.where(mask, True, np.nan)  # True stays True, False becomes np.nan
        axs[i].scatter(np.linspace(1, nsamp/sf, nsamp+1), mask_nan+ np.max(mean_df1[eye])+np.max(np.nanstd(last10arr1[:][:],axis=0)),color='black')    
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Pupil size (Au)')
        axs[i].legend()

    return fig



def getPupilDiamLast(allData,last10sDF = [],thrTime = 10000):
    """
        Function to get last 10s of any Data, differentiating between Tracked and Untracked Data.
    """

    last10sDF1 = []
    last10sDF2 = []

    for key in allData['data']['STORY_1'].keys():
        ltim = allData['data']['STORY_1'][key]['Gaze']['TimePoint'].iloc[-1]

        # Calculate the threshold for the last 10 seconds (10,000 ms)
        threshold = int(ltim) - thrTime
        # Select only rows where TimePoint is within the last 10 seconds
        last_10s_df = allData['data']['STORY_1'][key]['Gaze'][allData['data']['STORY_1'][key]['Gaze']['TimePoint'] >= threshold].reset_index()
        if 'KAROLINA' in key:
            last10sDF1.append(last_10s_df)
        else:
            last10sDF2.append(last_10s_df)
        last10sDF.append(last_10s_df)

    return last10sDF1, last10sDF2,last10sDF



def add_ds_trial_tracking(last10s_list, grand_df):
    """
        Cross-reference each trial DataFrame in last10s_list with grand_df to get DS, trialNum, and Tracking.
        Assumes order matches (i.e., last10s_list[i] corresponds to grand_df.iloc[i]).
    """
    
    out = []
    for i, df in enumerate(last10s_list):
        # Get subject/trial info as a DataFrame with 1 row
        currentGrandDF = grand_df.iloc[i][['Tracking', 'DS', 'trialNum']].copy()

        # Repeat info to match df's length
        repeated_info = pd.DataFrame([currentGrandDF.values] * len(df), columns=currentGrandDF.index)
        repeated_info.columns = [col[0] for col in repeated_info.columns]  # flatten
        repeated_info.index = df.index  # align index

        df = df.copy()
        df['SampleNum'] = np.arange(1, len(df) + 1)


        out.append(pd.concat([df, repeated_info], axis=1))

    return pd.concat(out, axis=0, ignore_index=True)


###### ---- (1) Testng MW_estimate vs Tracking/Untracking ---- ######
import statsmodels.api  as sm


def plotTrialTrack(m,name,grand_df,ax,titl):
  ### QUick Linear Model 
  quick_untracked_lm = lambda x: m.params[m.params.keys()[0]] + m.params[m.params.keys()[1]]+m.params[m.params.keys()[2]]*x 
  quick_tracked_lm = lambda x: m.params[m.params.keys()[0]] + m.params[m.params.keys()[2]]*x

  trialN = np.linspace(0,len(np.unique(grand_df['trialNum'])),len(np.unique(grand_df['trialNum']))+1);
  mw_est_utr = pd.Series(trialN).apply(quick_untracked_lm)
  mw_est_tr = pd.Series(trialN).apply(quick_tracked_lm)

  ax.set_title(titl)
  ax.plot(trialN,mw_est_tr,label="Tracked model slope")
  ax.plot(trialN,mw_est_utr,label="Tracked model slope")
  grTr = grand_df[[name,'trialNum','Tracking']].groupby(['Tracking','trialNum'])
  ax.fill_between(trialN, 
                  mw_est_tr - grTr.std().loc['TRACKED'][name], 
                    mw_est_tr+   grTr.std().loc['TRACKED'][name], 
                  color='darkblue', alpha=0.2, label='TRACKED (STD Envelope)')

  ax.scatter(grTr.mean().loc['TRACKED'].index,grTr.mean().loc['TRACKED'],label="Subj Mean Tracked",color = 'blue')

  ax.fill_between(trialN, 
                  mw_est_utr - grTr.std().loc['UNTRACKED'][name], 
                    mw_est_utr+   grTr.std().loc['UNTRACKED'][name], 
                  color='orangered', alpha=0.2, label='UNTRACKED (STD Envelope)')
  ax.scatter(grTr.mean().loc['UNTRACKED'].index,grTr.mean().loc['UNTRACKED'],label="Subj Mean Untracked",color = 'orange')
  ax.set_xlabel="Trial Number"
  ax.legend()
  
####### (3) ---- MW vs Gaze ----- #######


def plot_pupil_vs_mw(m, currentDf, ind, ax=None):
    """
    Plot Pupil vs MW_Estimate with linear fits for TRACKED and UNTRACKED groups.
    
    Parameters:
    - m: Fitted mixed effects model.
    - currentDf: DataFrame containing the data.
    - ind: Tuple indicating the current column (e.g., ('Pupil', 'Diameter')).
    - ax: Matplotlib Axes object. If None, a new figure will be created.
    """
    # Create a new figure and axis if no Axes object is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define linear functions for TRACKED and UNTRACKED
    linfunTr = lambda x: m.params[m.params.keys()[0]] + m.params[m.params.keys()[2]] * x
    linfunUnTr = lambda x: m.params[m.params.keys()[0]] + m.params[m.params.keys()[2]] * x + m.params[m.params.keys()[1]]

    # Scatter plot for TRACKED and UNTRACKED
    ax.scatter(currentDf[currentDf['Tracking'] == 'TRACKED']['MW_Estimate'],
               currentDf[currentDf['Tracking'] == 'TRACKED']['Pupil'],
               color='blue', label=f"Tracked (Coef: {m.params[m.params.keys()[1]]:1.4f})")
    ax.scatter(currentDf[currentDf['Tracking'] == 'UNTRACKED']['MW_Estimate'],
               currentDf[currentDf['Tracking'] == 'UNTRACKED']['Pupil'],
               color='orange', label=f"Untracked (pval: {m.pvalues[m.params.keys()[1]]:1.4f})")

    # Plot linear fits
    ax.plot(currentDf['MW_Estimate'], linfunTr(currentDf['MW_Estimate']), 'b--', label="Tracked Fit")
    ax.plot(currentDf['MW_Estimate'], linfunUnTr(currentDf['MW_Estimate']), 'orange', label="Untracked Fit")

    # Add labels, title, and legend
    ax.set_xlabel('MW Estimate')
    ax.set_ylabel(f"Pupil Measure: {ind[0]} // {ind[1]}")
    ax.set_title(f"{ind[0]} // {ind[1]}: (Coef: {m.params[m.params.keys()[2]]:1.4f} // P: {m.pvalues[m.params.keys()[2]]:1.4f})")
    ax.legend()

    # Add a suptitle if it's a standalone plot
    if ax is None:
        plt.suptitle(f"MW Estimate vs Pupil Measure")
        plt.show()


###### ---- Helpers ------ #####


def pad_columns(df: pd.DataFrame, depth: int, fill='') -> pd.DataFrame:
    """
    Ensure df.columns is a MultiIndex with exactly `depth` levels.
    Missing levels are right-padded with `fill`.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        # single-level → promote to MultiIndex
        df.columns = pd.MultiIndex.from_tuples([(c,) + (fill,) * (depth - 1) for c in df.columns])
    elif df.columns.nlevels < depth:
        df.columns = pd.MultiIndex.from_tuples(
            [tuple(col) + (fill,) * (depth - len(col)) for col in df.columns]
        )
    return df


######## ------ Analysis Checks ------- ########

def checkEntity(grand_df):
    trackedEntities = []
    ## Setup Working Directory and all necessery paths
    workingDir =  os.getcwd();
    path = os.path.split(workingDir)[0] + "\\data" #data path
    subjects = [f for f in os.listdir(path) if  not os.path.isfile(os.path.join(path, f))] # get all the filenames


    for filename  in subjects:

        with open(os.path.join(path, filename,filename + '_log.txt'), 'r') as fp:
            for line in fp:
                if "Current Named Entity" in line:
                    entityTracked = line.split("Current Named Entity: ")[1].strip()
                    trackedEntities.append(entityTracked)

                    break

    trackedEntities

    first_rows = grand_df.groupby('DS').first()
    first_rows.replace({'TRACKED': 'Karolina', 'UNTRACKED': 'Janek'},inplace=True)

    df =  pd.DataFrame([list(first_rows['Tracking']),trackedEntities],columns=first_rows['Tracking'],index=['df','original'])
    return df



def qualityChecks(grand_df,column):
    partiaclCheck1 = grand_df[grand_df['Checks']['']['goodRatio'+column] > 0.75]
    partiaclCheck2 = partiaclCheck1[partiaclCheck1['Checks']['']['goodPupRatio'+column] > 0.95]

    trmw1 = partiaclCheck1[partiaclCheck1['Tracking']=="TRACKED"].groupby(['Tracking','DS'])
    untrmw1  = partiaclCheck1[partiaclCheck1['Tracking']=="UNTRACKED"].groupby(['Tracking','DS'])

    trmw2 = partiaclCheck2[partiaclCheck2['Tracking']=="TRACKED"].groupby(['Tracking','DS'])
    untrmw2  = partiaclCheck2[partiaclCheck2['Tracking']=="UNTRACKED"].groupby(['Tracking','DS'])


    return partiaclCheck1,partiaclCheck2,trmw1,untrmw1,trmw2,untrmw2


def linearModelChecks(m,grand_df):
    # trial-level diagnostics
    resid   = m.resid                   # raw (conditional) residuals
    fitted  = m.fittedvalues            # conditional fitted values
    std_res = resid / np.sqrt(m.scale)  # ≈ studentised residuals
    fig,axs = plt.subplots(2,1,figsize=(14, 8))
    plt.suptitle("Checks for a Linear Model")

    # (1) Homoscedascisity
    axs[0].scatter(fitted, std_res, alpha=0.6, s=15)
    axs[0].axhline(0, linestyle='--')
    axs[0].set_xlabel("Fitted value")
    axs[0].set_ylabel("Studentised residual")
    axs[0].set_title("Residuals vs fitted")

    # (2) 
    sm.qqplot(std_res, line="45", fit=True, ax=axs[1])
    axs[1].set_title("Normal Q–Q")

    # (3) Colinearity:
    from patsy import dmatrices

    formula_fixed = "MW_Estimate ~ Tracking + trialNum"   # your fixed part
    y, X = dmatrices(formula_fixed, data=grand_df[['DS','trialNum','Tracking','MW_Estimate']], return_type="dataframe")
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns
    )
    print(vif)

     # Drop the intercept from display (its VIF is usually not meaningful)
    vif_display = vif.drop('Intercept', errors='ignore')
    
    # Convert to multiline string
    vif_text = "\n".join([f"{k}: {v:.2f}" for k, v in vif_display.items()])
    vif_text = f"VIFs\n{vif_text}"

    axs[1].text(
        0.8, 0.5, vif_text,
        transform=axs[1].transAxes,
        verticalalignment='top',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8)
    )
    plt.show()
