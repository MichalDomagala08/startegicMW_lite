import re

def testTiming(log,outputControll):
    sumDuration = 0
    outputControll.write("Check Timing:")
    for i in range(len(log)-1):
        currentTime =  log[i+1][1]
        currentRT =  log[i+1][2]
        currentDuration =  log[i+1][3]

        previousEvent =  log[i][0]
        previousTime = log[i][1]

        # CHECK Probe validity:
        if previousEvent == "BEGIN":

            if currentTime - currentDuration - previousTime <0.005:
                outputControll.write(f"Error Below 5 ms: for PROBE in {i+1} and BEGIN in {i} ({ currentTime - currentDuration - previousTime :.4f})")
            else:
                Warning(f"The mismatch between BEGIN in {i} and PROBE in {i+1} is greater than 5 ms!")

        elif previousEvent =="PROBE":

            if currentTime - currentRT - previousTime <0.005:
                outputControll.write(f"Error Below 5 ms: for KEYPRESS in {i+1} and PROBE in {i} ({ currentTime - currentRT - previousTime :.4f})")
            else:
                Warning(f"The mismatch between PROBE in {i} and KEYPRESS in {i+1} is greater than 5 ms!")

            sumDuration += currentDuration+currentRT
    outputControll.write(f"Difference Between Summed StoryTimes and Summed Event Clock: {log[-1][1]-sumDuration:.4f}")


import pandas as pd
def prepareEyeTrackingTiming(eyetrackerFile,outputControll):
    with open(eyetrackerFile, 'r') as file:
        msg_lines = [
            line.strip() for line in file 
            if re.match(r"^MSG\s+\d+", line) and 
               not any(exclude in line for exclude in ["!CAL", "VALIDATE", "!MODE", "ELC", "CAMERA", "THRESHOLDS", "GAZE", "RECC"])
        ]
   

        table = [re.split(r'[\t]+', line) for line in msg_lines]
      
        for i in range(len(table)):
            if i == 0:
                table[i].append(0)  # First row has 0 duration
            else:
                table[i].append(float(table[i][1]) - float(table[i - 1][1]))


        df = pd.DataFrame(table,columns = ["MSG","Time","Event","Type","Status","Duration"])
        moveTime =  df.pop("Time");
        df.pop("MSG")
        df.insert(3, "Time", moveTime)

        outputControll.write("Extracted Event Data: ")
        outputControll.write(df.to_string())
        outputControll.write(" ")

        #Iterate through rows and compute things: 
        nextStory = 0
        i = 0
        eyeTrackingLogs = {"story1": pd.DataFrame(),
                           "story2": pd.DataFrame()}
        for j in [0,1]:
            Event, StoryTime, RT, PartDuration = [], [], [], []
            while i <= len(df):
                row = df.iloc[i]
                ## Start Story 
                if row['Event'] == 'PART' and row['Status'] =='BEG':
                    Event.append("BEGIN")
                    if i == 1 or nextStory:
                        StoryTime.append(float(row['Duration'])/1000)
                        begTime = i
                        nextStory= 0
                    else:
                        StoryTime.append((float(row['Time'])-float(df.iloc[begTime]['Time'])+ float(df.iloc[begTime]['Duration'])/1000)/1000)
                    RT.append(0)
                    PartDuration.append(0)
                elif row['Event'] == 'PART' and row['Status'] =='END':
                    Event.append("PROBE")
                    StoryTime.append((float(row['Time'])-float(df.iloc[begTime]['Time'])+ float(df.iloc[begTime]['Duration']))/1000)
                    RT.append(0)
                    PartDuration.append(float(row['Duration'])/1000)
                elif row['Event'] =='KEYPRESS':
                    Event.append("KEYPRESS")
                    StoryTime.append((float(row['Time'])-float(df.iloc[begTime]['Time'])+ float(df.iloc[begTime]['Duration']))/1000)
                    RT.append(float(row['Duration'])/1000)
                    PartDuration.append(float(df.iloc[i-1]['Duration'])/1000)
                elif row['Event'] == 'RECALL': #Break on the nearest recall
                    i +=2
                    nextStory = 1
                    break;
                i +=1;

            eyeTrackingLogs[list(eyeTrackingLogs.keys())[j]] = pd.DataFrame(list(map(list, zip(*[Event,StoryTime,RT,PartDuration]))),columns=["Event","StoryTime","RT","partDuration"])
            outputControll.write(eyeTrackingLogs[list(eyeTrackingLogs.keys())[j]].to_string())

        return eyeTrackingLogs
    

def compareEyeTrackingWithBeh(eyeTrackingLogs,logStory1,logStory2,outputControll):

    ### Print Differences 
    df_diff1 = eyeTrackingLogs['story1'] .copy()
    df_diff1[["StoryTime", "RT", "partDuration"]] = eyeTrackingLogs['story1'][["StoryTime", "RT", "partDuration"]] - logStory1[["StoryTime", "RT", "partDuration"]]
    
    outputControll.write("Adjusted EyeTracking DF for Story 1: ")
    outputControll.write(df_diff1.to_string())


    df_diff2 = eyeTrackingLogs['story2'] .copy()
    df_diff2[["StoryTime", "RT", "partDuration"]] = eyeTrackingLogs['story2'][["StoryTime", "RT", "partDuration"]] - logStory2[["StoryTime", "RT", "partDuration"]]
    outputControll.write("Adjusted EyeTracking DF for Story 2: ")
    outputControll.write(df_diff2.to_string())

    ## get Numeric data: 

    numDiffData1 = df_diff1.select_dtypes(include='number').to_numpy()
    numDiffData2 = df_diff2.select_dtypes(include='number').to_numpy()
    def diffr(numDiffData,df_diff):
        for i in range(numDiffData.shape[1]):
            for j in range(numDiffData.shape[0]):
                if numDiffData[j,i] > 0.005:
                    outputControll.write(f"   WARNING! Exceeding 5ms delay threshold between EyeLink and behavior in row {j} ({df_diff.iloc[j]['Event']}) on {df_diff.columns.values.tolist()[i]} (Delay: {numDiffData1[j,i]:.5f})")
                else:
                    outputControll.write(f"   Delay is in norm for row {j} ({df_diff.iloc[j]['Event']}) on {df_diff.columns.values.tolist()[i]} (Delay: {numDiffData[j,i]:.5f})")

    outputControll.write("\n Differnce in Story 1 Check: ")
    diffr(numDiffData1,df_diff1)
    outputControll.write("\n Differnce in Story 2 Check: ")
    diffr(numDiffData2,df_diff2)