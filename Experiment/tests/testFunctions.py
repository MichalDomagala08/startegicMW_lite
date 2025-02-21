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


def testEyeTrackerTiming(eyetrackerFile,outputControll):
    with open(eyetrackerFile, 'r') as file:
        msg_lines = [line.strip() for line in file if re.match(r"^MSG\s+\d+", line)]

        for msgstart,line in enumerate(msg_lines):
            if "STORY 1" in line:
                break

        msg_lines_true = msg_lines[msgstart:]

        for i in msg_lines_true:
            print(i)