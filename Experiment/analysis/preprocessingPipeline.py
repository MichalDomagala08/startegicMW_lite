"""
This script is used to preprocess the data for the experiment from each Individual Subject
It includes functions to load the data, preprocess it, and save the preprocessed data.

"""
#===============================================================#
#### ---> GET SETUP DATA AND ALL NECESSERY DEPENDENCIES <--- ####
#===============================================================#

######## --> LIBRARIES AND PATHS  <-- ########

### Libraries
import sys
import numpy as np
import os 
import pandas as pd
from analysisFunctions import * # Get all Functions for analysis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import openpyxl
import pickle

### Setup Working Directory and all necessery paths
workingDir =  os.path.dirname(os.path.abspath(__file__));
print("Working Directory: ", workingDir)
print("Current Directory: ", os.path.dirname(os.path.abspath(__file__)))
path = os.path.split(workingDir)[0] + "\\data" #data path


######## --> SETUP ALL PARAMETERS <-- ########

### Setup all general parameters:
verbose = 1 # Verbose mode
analysisName = 'TEST_'
### Setup all cleaning parameters

# General parameters
paramsDict = {};
paramsDict['verbose'      ]  = 1;    # Whether we want to print plots or not 
paramsDict['resRate'      ]  = 100;  # Resampling Rate

# Centering parameters
paramsDict['dgvCent'      ]  = 5;    # How many degrees of visual angle are considered as a Centered Gaze

# Blink Interpolation parameters
paramsDict['blinkInt'     ]  = 0;    #Blink interpolation merhod: 0-linear; 1-cubic
paramsDict['maxblength'   ]  = 500;  # Maximum length of a missing data event - for centering computation 
paramsDict['blkboundry'   ]  = 75;   # TIme in miliseconds to extent the range around faulty periods to remove-count (CHECK! Whether you are considering Padding in thresholding blinks)
paramsDict['blinkbegbound']  = 1000; # OLD - the time in ms after exp onset that the blinks are considered

# Smoothing and clusterinf 
paramsDict['smoothwin'    ]  = 10;   # Window for smoothing (in samples)
paramsDict['maxcluslen'   ]  = 2000; # Minimum length  (in ms) of a cluster to be smoothed 
paramsDict['mingaplen'    ]  = 50;  # Maximum gap (in ms) between timestamps within a cluster.

# Pupil Diameter Computation 
paramsDict['pupMode'      ]  = 'mean' # mean or std computation to assess trial Pupil Size (TBD - for now both are used) 
paramsDict['trialOffset'  ]  = 10000  # How many ms before the trial offset do we compute our measure (10s default)

# Baseline Pupil Computation
paramsDict['baseOffset'] = 0 # How many ms after the trials onset will be considered a baseline:  0 if you do not want baseline
### Setup all testing parameters 


### Subject Choice
begID = 0;
endID = 14

analysisName = f"TEST_res{paramsDict['resRate']}_dgv{paramsDict['dgvCent']}_bll{paramsDict['maxblength']}_blb{paramsDict['blkboundry']}_smw10_smL{paramsDict['maxcluslen']}_swG{paramsDict['mingaplen']}_trO{paramsDict['trialOffset']}_trB{paramsDict['baseOffset']}"
# Get all of the IDs for subject froma "data" folder
subjects = [f for f in os.listdir(path) if  not os.path.isfile(os.path.join(path, f))] # get all the filenames

### Create a current Analysis Pipeline
if os.path.isdir(os.path.join(workingDir,analysisName,'individuals')) == 0:
        os.makedirs(os.path.join(workingDir,analysisName,'individuals'))
allList = [];

### Createa an excel for data:
filepath = os.path.join(workingDir,analysisName,'individuals',"analysis_results.xlsx")
wb = openpyxl.Workbook()
wb.save(filepath)

### Setup All Data Structures and PDFs 

# General Raw Plots:
output_dir = os.path.join(workingDir,analysisName, "individuals") #output path

pdf_path_raweyegaze = os.path.join(output_dir, "RawEyeGaze.pdf")
pdf_raweyegaze = PdfPages(pdf_path_raweyegaze)
pdf_path_raweyekde = os.path.join(output_dir, "RawEyeKDE.pdf")
pdf_raweyekde = PdfPages(pdf_path_raweyekde)
pdf_path_rawpupil = os.path.join(output_dir, "RawPupil.pdf")
pdf_rawpupil  = PdfPages(pdf_path_rawpupil)

pdf_path_gazecheck = os.path.join(output_dir, "gazeCheckPars.pdf")
pdf_gazecheck  = PdfPages(pdf_path_gazecheck)

pdf_path_mwHist = os.path.join(output_dir, "mwHist.pdf")
pdf_mwHist  = PdfPages(pdf_path_mwHist)

pdf_path_scatres = os.path.join(output_dir, "scatterResults.pdf")
pdf_scatres  = PdfPages(pdf_path_scatres)

#=========================================#
##### --->    EXTRACTION LOOP    <--- #####
#=========================================#

for subj in range(begID,endID+1):

    ### SUBJECT SPECIFIC SETUPS AND VARIABLES 
    filename = subjects[subj]

    # Open Log File
    if os.path.isdir(os.path.join(workingDir,analysisName,'individuals',filename)) == 0:
        os.makedirs(os.path.join(workingDir,analysisName,'individuals',filename))
    log_file = open(os.path.join(workingDir,analysisName,'individuals',filename,filename+"_preprocLog.txt"),'w+')
    log_fileSt = open(os.path.join(workingDir,analysisName,'individuals',filename,filename+"_statLog.txt"),'w+')

    # Get the tracked entity name and display parameters (TBD)
    with open(os.path.join(path, filename,filename + '_log.txt'), 'r') as fp:
        for line in fp:
            if "Current Named Entity" in line:
                entityTracked = line.split("Current Named Entity: ")[1].strip()
                break

    
    log_file.write(" "*15+"="*58 + "\n")
    log_file.write(" "*15+"="*5+ " "*15+" PREPROC LOG FILE"+ " "*15+"="*5+ " \n")
    log_file.write(" "*15+"="*58 + "\n\n")
    log_file.write("   General Info: " + filename + "\n")
    print("#-------- Current Subject: ", filename, " --------#")
    log_file.write("      Participant ID: " + filename + "\n")
    log_file.write("      Tracked Entity: " + entityTracked + "\n\n")
    print("Tracked Entity:", entityTracked, " \n")



    #==========================================#
    #### -->   LOAD AND PREPARE DATA   <--- ####
    #==========================================#

    log_file.write("\n===# Load and Prepare Data #=== \n\n")

    log_file.write("    Loading Participant Data... ")

    [own,rawEye, events, gazeCoords] = loadParticipant(path, name=filename) # Gets my RawEye Data as well as events and gaze Coordination
    RawDF = pd.DataFrame(rawEye,columns=["TimePoint",'LeftX','LeftY','LeftPupil','RightX','RightY','RightPupil']);
    log_file.write("done \n")

    # Get Events
    log_file.write("    Getting Events...           ")
    fixationDF,saccadesDF,blinkDF = getEvent(events)
    log_file.write("done \n")

    # Change Timing so it will be relative from the start of the experiment
    log_file.write("    Changing Timing....         ")
    RawDF,fixationDF,saccadesDF,blinkDF,own = changeTiming(RawDF,fixationDF,saccadesDF,blinkDF,own)
    log_file.write("done \n")

    # Parse your own events getting a Wide format getting only PARTS with their Timing  (Maybe Write Checks?)
    log_file.write("    Getting Own Events...       ")
    event_dfs = parse_events(own)
    event_dfs = rename_parts(event_dfs) # Be sure that the numeration for each entity part is Correct! 
    log_file.write("done \n")
    if len(event_dfs) != 40:
        log_file.write("    WARNING! the events are not whole!       ")
    log_file.write("\n            Events DataFrame:       \n")

    log_file.write(event_dfs.describe().to_string())
    # Contains All our Data but now Comnvienientl Split for a stories 
    log_file.write("\n\n    Splitting by events...      ")
    split_data = split_dataframes_by_events(RawDF, fixationDF, saccadesDF, blinkDF, event_dfs)
    log_file.write("done \n")

    if verbose == 2: # Print the split data if You want to see them
        for splt in split_data.keys():
            for event_name, df in split_data[splt]['RawDF'].items():
                if verbose:
                    print(df)
            for event_name, df in split_data[splt]['fixationDF'].items():
                print(f"    fixationDF for {event_name}:")
                if verbose:
                    print(df)
            for event_name, df in split_data[splt]['saccadesDF'].items():
                print(f"    saccadesDF for {event_name}:")
                if verbose:
                    print(df)
            for event_name, df in split_data[splt]['blinkDF'].items():
                print(f"    blinkDF for {event_name}:")
                if verbose:
                    print(df)

   
    #==========================================#
    #### -->   GENERAL DATA OVERVIEW   <--- ####
    #==========================================#


     ### Get yourself only Trial Data (as not to get contaminated)
    [firstEntityRaw, secondEntityRaw] = splitEntities(RawDF, event_dfs)
    [firstEntityBlinks, secondEntityBlinks] = splitEntities(blinkDF, event_dfs)
    [firstEntitySaccades, secondEntitySaccades] = splitEntities(saccadesDF, event_dfs)
    [firstEntityFixation, secondEntityFixation] = splitEntities(fixationDF, event_dfs)    

    allEntities = pd.concat([firstEntityRaw, secondEntityRaw], ignore_index=True)
    allEntities.sort_values(by='TimePoint', inplace=True)

    log_file.write("\n===# Stats and Checks for Experiment Relevant only Data #=== \n\n ")

    [RawDFPupilTrue,_] = GeneralChecks(allEntities, blinkDF, saccadesDF, fixationDF, gazeCoords)

    if verbose == 1:
        fig1 = eyeRawKDE(RawDF,titl=filename)
        pdf_raweyekde.savefig(fig1)
        plt.close(fig1)

        fig2 =plotRawGaze(RawDF,gazeCoords,titl=filename)
        pdf_raweyegaze.savefig(fig2)
        plt.close(fig2)

        fig3 =plotPupilTimecourse(RawDFPupilTrue, f"{filename}: Raw Pupil Data with Blinks", blinkDF, saccadesDF, fixationDF, chooseViz='001')
        pdf_rawpupil.savefig(fig3)
        plt.close(fig3)

    ### Quality Metrics for individual participant
    log_file.write("\n      ===# GeneralCheck Parameters All Length #=== \n\n ")
    gazeDiff, pupilDiff, saccDiff, blinkDiff, eyeposDiff, nogazeRate,_,_ = qualiryMetrics([firstEntityRaw, secondEntityRaw], [firstEntitySaccades, secondEntitySaccades], [firstEntityBlinks, secondEntityBlinks], gazeCoords, log_file)

    def chooseFirstSecondHalf(firstEntityRaw):

        if 'TimePoint' in firstEntityRaw.columns:
            firstEntityRaw1       =  firstEntityRaw[firstEntityRaw['TimePoint']  < np.median(firstEntityRaw['TimePoint'])]
            firstEntityRaw2       =  firstEntityRaw[firstEntityRaw['TimePoint'] > np.median(firstEntityRaw['TimePoint'])]

        elif 'Beg' in firstEntityRaw.columns:
            firstEntityRaw1       =  firstEntityRaw[firstEntityRaw['Beg']  < np.median(firstEntityRaw['Beg'])]
            firstEntityRaw2       =  firstEntityRaw[firstEntityRaw['Beg'] > np.median(firstEntityRaw['Beg'])]

        return firstEntityRaw1, firstEntityRaw2


    firstEntityRaw1, firstEntityRaw2 = chooseFirstSecondHalf(firstEntityRaw)
    secondEntityRaw1, secondEntityRaw2 = chooseFirstSecondHalf(secondEntityRaw)
    firstEntityBlinks1, firstEntityBlinks2 = chooseFirstSecondHalf(firstEntityBlinks)
    secondEntityBlinks1, secondEntityBlinks2 = chooseFirstSecondHalf(secondEntityBlinks)
    firstEntitySaccades1, firstEntitySaccades2 = chooseFirstSecondHalf(firstEntitySaccades)
    secondEntitySaccades1, secondEntitySaccades2 = chooseFirstSecondHalf(secondEntitySaccades)

    log_file.write("\n      ===# GeneralCheck Parameters 1st Half #=== \n\n ")
    print("\n      ===# GeneralCheck Parameters 1st Half #=== \n\n ")
    gazeDiff1, pupilDiff1, saccDiff1, blinkDiff1, eyeposDiff1, nogazeRate1,_,_  = qualiryMetrics([firstEntityRaw1, secondEntityRaw1],
                                                                                                [firstEntitySaccades1, secondEntitySaccades1], 
                                                                                                [firstEntityBlinks1, secondEntityBlinks1], gazeCoords, log_file)

    log_file.write("\n      ===#  GeneralCheck Parameters 2st Half  #=== \n\n ")
    print("\n      ===# GeneralCheck Parameters 2nd Half #=== \n\n ")
    gazeDiff2, pupilDiff2, saccDiff2, blinkDiff2, eyeposDiff2, nogazeRate2,_,_  = qualiryMetrics([firstEntityRaw2, secondEntityRaw2], 
                                                                                                [firstEntitySaccades2, secondEntitySaccades2], 
                                                                                                [firstEntityBlinks2, secondEntityBlinks2], gazeCoords, log_file)
    #===============================================#
    #### -->   PREPROCESSING AND PLOTTING   <--- ####
    #===============================================#

    # Other Checks:
    storyIndCheck = {}

    # Define PDF paths
    pdf_path_downsampled = os.path.join(output_dir,filename, "Downsampled_Pupil_Data.pdf")
    pdf_path_screen_only = os.path.join(output_dir,filename, "Screen_Only.pdf")
    pdf_path_blink_rejection = os.path.join(output_dir,filename, "After_Pupil_Artefacts_Rejection.pdf")
    pdf_path_outlier_rejection = os.path.join(output_dir,filename, "After_Outlier_Rejection.pdf")
    pdf_path_centered = os.path.join(output_dir,filename, "CenteredPupil.pdf")
    pdf_path_nan_rejection = os.path.join(output_dir,filename,"NaN_rejected_Pupil.pdf")
    # Open PdfPages objects for each PDF

    pdf_downsampled = PdfPages(pdf_path_downsampled)
    pdf_center_only = PdfPages(pdf_path_screen_only)
    pdf_blink_rejection = PdfPages(pdf_path_blink_rejection)
    pdf_outlier_rejection = PdfPages(pdf_path_outlier_rejection)

    pdf_nan_rejection = PdfPages(pdf_path_nan_rejection)
    pdf_smoothed = PdfPages(pdf_path_centered)
    log_file.write("\n===# Preprocessing the Data #=== \n\n ")
    log_file.write(f"    Preproc Parameters: General: \n")
    log_file.write(f"        Verbose:              {paramsDict['verbose']}\n")
    log_file.write(f"        Resampling Rate:      {paramsDict['resRate']}\n")
    log_file.write(f"        DGV from Center:      {paramsDict['dgvCent']}\n\n")
    log_file.write(f"    Preproc Parameters: Blinks: \n")
    log_file.write(f"        Blink Interpolation:  {paramsDict['blinkInt']}\n")
    log_file.write(f"        Maximum Blink Length: {paramsDict['maxblength']}\n")
    log_file.write(f"        Blink Interp Bound:   {paramsDict['blkboundry']}\n")
    log_file.write(f"        Blink Begining Bound: {paramsDict['blinkbegbound']}\n\n")
    log_file.write(f"    Preproc Parameters: Smooth: \n")
    log_file.write(f"        Window size:          {paramsDict['smoothwin']}\n")
    log_file.write(f"        Minimal Cluster len:  {paramsDict['maxcluslen']}\n")
    log_file.write(f"        Maximal Gap Length:   {paramsDict['mingaplen']}\n\n")
    # Iterate through events of split_data
    data = {};
    finalDataStruct = {}
    resultsL = np.zeros((len(split_data['STORY_1']['RawDF']), 5))
    resultsR = np.zeros((len(split_data['STORY_1']['RawDF']), 5))

    ### Loop thtough Stories X Parts: 
    iterCounter = 0;
    for story, data_dict in split_data.items():
        finalDataStruct[story] = {}
        data[story] = {}
        storyIndCheck[story] = {}
        storyIndCheck[story]['goodRatioLeft']      = []
        storyIndCheck[story]['goodRatioRight']     = []
        storyIndCheck[story]['goodPupRatioLeft']   = []
        storyIndCheck[story]['goodPupRatioRight']  = []
        storyIndCheck[story]['gazeDiff']    = []
        storyIndCheck[story]['gazeRate']    = []
        storyIndCheck[story]['pupilDiff']   = []
        storyIndCheck[story]['meanEyeDiff'] = []
        storyIndCheck[story]['saccadeRate'] = []
        storyIndCheck[story]['blinkRate'] = []
        for part, RawDF in data_dict['RawDF'].items():
            print(f"\nAnalysing: {story} // {part} ")
            log_file.write(f"    ### Analysing: {story} // {part} ###\n")
            data[story][part] = {}

            blinkDF = data_dict['blinkDF'][part]
            saccadesDF = data_dict['saccadesDF'][part]


            finalData = preprocessingPipeline(blinkDF,RawDF,saccadesDF,gazeCoords,story,part,log_file=log_file,pdfs=[pdf_downsampled,pdf_center_only,pdf_blink_rejection ,pdf_outlier_rejection ,pdf_nan_rejection,pdf_smoothed],
                                            verbose=paramsDict['verbose'],interp_type=paramsDict['blinkInt'],interpBoundary=paramsDict['blkboundry'],maxBlinkDur=paramsDict['maxblength'],resampleRate=paramsDict['resRate'],
                                            dgvCenter=paramsDict['dgvCent'],smoothwin=paramsDict['smoothwin'], min_cluster_duration=paramsDict['maxcluslen'],max_gap_duration=paramsDict['mingaplen'])
            
            log_file.write("\n        Final Data Stats: \n")
            log_file.write(finalData.describe().to_string().replace('\n', '\n\t\t').join(['\t\t', ''])) # Adds two Tabulations!
            log_file.write("\n\n")
            log_file.write("        Sanity Checks on Final Data: \n")
            log_file.write(f"            Number of Non-missing Data events: {len(finalData['LeftPupil'].dropna())} ({len(finalData['LeftPupil'].dropna())/len(finalData)} %)\n")
            gazeD, _, _, _, _, _,pupil_mean,mean_eye_diff  = qualiryMetrics([finalData], [saccadesDF], [blinkDF], gazeCoords, log_file)
            

            ### Compute mean of last 10s of the data
            meanPupilL = finalData[(finalData['TimePoint'] >= finalData['TimePoint'].iloc[-1] - paramsDict['trialOffset'  ])]['LeftPupil']
            meanPupilR = finalData[(finalData['TimePoint'] >= finalData['TimePoint'].iloc[-1] - paramsDict['trialOffset'  ])]['RightPupil']
 
            ### Correct for Baseline if YOu want to
            if paramsDict['baseOffset']:
                meanPupilL = meanPupilL - finalData[(finalData['TimePoint'] <= finalData['TimePoint'].iloc[0] +  paramsDict['baseOffset'  ])]['LeftPupil'].mean()
                meanPupilR = meanPupilR - finalData[(finalData['TimePoint'] <= finalData['TimePoint'].iloc[0] +  paramsDict['baseOffset'  ])]['RightPupil'].mean()

            last10 = finalData[(finalData['TimePoint'] >= finalData['TimePoint'].iloc[-1] - paramsDict['trialOffset'  ])]
            gazeDifference = np.mean(np.sqrt((last10['LeftX'] - last10['RightX'])**2 + (last10['LeftY'] - last10['RightY'])**2))

            resultsL[iterCounter,:] =  [meanPupilL.mean(), meanPupilL.std(),np.nanmean(np.diff(meanPupilL)), event_dfs[event_dfs["Part"] == part]['key'].values[0],gazeDifference]
            resultsR[iterCounter,:] =  [meanPupilR.mean(), meanPupilR.std(),np.nanmean(np.diff(meanPupilR)), event_dfs[event_dfs["Part"] == part]['key'].values[0],gazeDifference]

            iterCounter+=1;


            finalDataStruct[story][part] = finalData;
            data[story][part]['Gaze']     =finalData;
            data[story][part]['Blinks']   =blinkDF;
            data[story][part]['Saccades'] =saccadesDF;

            storyIndCheck[story]['saccadeRate'].append(len(saccadesDF)/(finalData['TimePoint'].iloc[-1] - finalData['TimePoint'].iloc[0])/1000)
            storyIndCheck[story]['blinkRate'].append(len(blinkDF)/(finalData['TimePoint'].iloc[-1] - finalData['TimePoint'].iloc[0])/1000)

            storyIndCheck[story]['goodRatioLeft'].append(len(finalData['LeftPupil'].dropna())/len(finalData))
            storyIndCheck[story]['goodRatioRight'].append(len(finalData['LeftPupil'].dropna())/len(finalData))
            storyIndCheck[story]['goodPupRatioLeft'].append(len( finalData[(finalData['TimePoint'] >= finalData['TimePoint'].iloc[-1] - paramsDict['trialOffset'  ])]['LeftPupil'].dropna())/len( finalData[(finalData['TimePoint'] >= finalData['TimePoint'].iloc[-1] - paramsDict['trialOffset'  ])]))
            storyIndCheck[story]['goodPupRatioRight'].append(len( finalData[(finalData['TimePoint'] >= finalData['TimePoint'].iloc[-1] - paramsDict['trialOffset'  ])]['LeftPupil'].dropna())/len( finalData[(finalData['TimePoint'] >= finalData['TimePoint'].iloc[-1] - paramsDict['trialOffset'  ])]))
            
            storyIndCheck[story]['gazeDiff'].append(np.mean(gazeD[0,:]))
            zeroPupil = RawDF.query("LeftPupil == 0 or RightPupil == 0")
            storyIndCheck[story]['gazeRate'].append( len(zeroPupil) / len(RawDF))
            storyIndCheck[story]['pupilDiff'].append(pupil_mean)
            storyIndCheck[story]['meanEyeDiff'].append(mean_eye_diff)

    # Close PdfPages objects
    pdf_downsampled.close()
    pdf_center_only.close()
    pdf_blink_rejection.close()
    pdf_outlier_rejection.close()
    pdf_smoothed.close()
    pdf_nan_rejection.close()
    print("Processing complete. PDFs saved.")
    log_file.write("\n===# Preprocessing Completed Successfully #===\n")
    log_file.write(f"  Total trials processed: {iterCounter}\n")
    log_file.write(f"  FinalDataStruct contains stories: {list(finalDataStruct.keys())}\n")
    log_file.write("===# End of Preprocessing Section #===\n\n")
    log_file.close() # Closing the preprocessing log file)

    ### Create a DF:

    storyCheckDF = pd.DataFrame(storyIndCheck['STORY_1'])
    print(storyIndCheck)

    #============================================================#
    #### -->    AFTER SEGMENTATION DATA QUALITY CHECK    <--- ####
    #============================================================#
    
      
    log_fileSt.write(" "*15+"="*58 + "\n")
    log_fileSt.write(" "*15+"="*5+ " "*15+" STATISTIC LOG FILE"+ " "*15+"="*5+ " \n")
    log_fileSt.write(" "*15+"="*58 + "\n\n")
    log_fileSt.write("   General Info: " + filename + "\n")
    log_fileSt.write("      Participant ID: " + filename + "\n")
    log_fileSt.write("      Tracked Entity: " + entityTracked + "\n\n")


    tableData = {
        ('MW Estimate', '', ''): resultsL[:, 3],  # MW Estimate is shared across all
        ('GazeDifference', '', ''): resultsL[:, 4],  # Difference of Right-Left Gaze

        ('Pupil Diameter', 'Mean', 'Left Eye'): resultsL[:, 0],
        ('Pupil Diameter', 'Mean', 'Right Eye'): resultsR[:, 0],
        ('Pupil Diameter', 'Std', 'Left Eye'): resultsL[:, 1],
        ('Pupil Diameter', 'Std', 'Right Eye'): resultsR[:, 1],
        ('Pupil Diameter', 'diff', 'Left Eye'): resultsL[:, 2],
        ('Pupil Diameter', 'diff', 'Right Eye'): resultsR[:, 2],
        ('Tracking','','') : ['TRACKED' if entityTracked in ent else 'UNTRACKED' for ent in event_dfs['Part']]
    }
    resultsDF = pd.DataFrame(tableData)
    resultsDF.columns = pd.MultiIndex.from_tuples(resultsDF.columns)
    print(resultsDF)


    ### Plotting Gaze X Quwality Metrics through time ( Left-Right Difference; )
    fig1 = plot_gaze_check(storyIndCheck['STORY_1'],filename,entityTracked)
    pdf_gazecheck.savefig(fig1)
    plt.close(fig1)

    fig2 =  mwHist(resultsDF,filename)
    pdf_mwHist.savefig(fig2)
    plt.close(fig2)

    fig3 = scatterResults(resultsL[:, 3],[resultsL[:, 0],resultsR[:, 0],resultsL[:, 1],resultsR[:, 1],resultsL[:, 2],resultsR[:, 2]],filename)
    pdf_scatres.savefig(fig3)
    plt.close(fig3)

    ### Wilcoxon Test of Tracked/Untracked Difference: In MW estimation 
    statsDF = wilcLoop(resultsDF,entityTracked,verbose=verbose,log_file=log_fileSt)
    
    # Log summary of Wilcoxon results (unpacking the MultiIndex) 
    log_fileSt.write("===# Wilcoxon Test Summary #===\n")
    log_fileSt.write(statsDF.to_string())
    log_fileSt.write("===# End Wilcoxon Summary #===\n\n")


    # Export DataFrame to Excel as a new sheet
    def save_df_to_excel(df, filename, sheet_name="Stats"):
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    # Save stats_df to Excel
    save_df_to_excel(statsDF, os.path.join(workingDir,analysisName,'individuals','analysis_results.xlsx'), sheet_name=filename)
    #===============================================#
    #### -->    SAVING PREPROCESSED DATA    <--- ####
    #===============================================#

    subjectInfo = {
        'filename': filename,
        'chEntity': 'Karolina'
    }
    checkInfo = {
        'gazeD'   : [gazeDiff1,gazeDiff2],
        'pupilD ' : [pupilDiff1,pupilDiff2],
        'nogazeR ': [nogazeRate1,nogazeRate2],
        'saccD'   : [saccDiff1,saccDiff2],
        'blinkD'  : [blinkDiff1, blinkDiff2],
        'eyeposD' : [eyeposDiff1, eyeposDiff2],
        'storyIndCheck': storyCheckDF
    }

    finalData = {
        'data':         data,
        'params':       paramsDict,
        'subjectInfo':  subjectInfo,
        'checkInfo':    checkInfo,
        'pupilMW':      resultsDF,
        'statistics':   statsDF}
    
    with open(os.path.join(workingDir,analysisName,'individuals',filename,filename+"_preprocessed.pickle"), 'wb') as handle:
        pickle.dump(finalData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log_fileSt.write(f"Saved pickle to {filename}_preprocessed.pickle ({os.path.getsize(os.path.join(workingDir,analysisName,'individuals',filename,filename+'_preprocessed.pickle'))/1e6:.2f} MB)\n")
    log_fileSt.write("===# End of Statistics Logging #===\n")
    log_fileSt.close() # Closing the preprocessing log file)

    allList.append(finalData)


with open(os.path.join(workingDir,analysisName,'individuals',"preprocessed.pickle"), 'wb') as handle:
    pickle.dump(allList, handle, protocol=pickle.HIGHEST_PROTOCOL)

pdf_raweyegaze.close()
pdf_raweyekde.close()
pdf_rawpupil.close()
pdf_gazecheck.close()
pdf_scatres.close()
pdf_mwHist.close()
print("DONE!")
# Setup Log and Subject Specific LOG file 

