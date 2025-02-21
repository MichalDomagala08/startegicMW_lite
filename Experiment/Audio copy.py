import pyaudio
import pygame
import wave
import threading
import time
from random import uniform
import sys

class audioTrial:

    def __init__(self, audioPath, storyPartDict, font, screen,currentStage,nextStage, output_control,chunk_size=256,verbose=0,tpDistribution=(5,10),tpTimingMode='semirandom'):

        ### Screen
        self.font = font
        self.screen = screen
        self.width = screen.get_size()[0]
        self.height = screen.get_size()[1]

        ### Audio Stream
        self.audio_file = wave.open(audioPath, 'rb')
        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=self.p.get_format_from_width(self.audio_file.getsampwidth()),
                                  channels=self.audio_file.getnchannels(),
                                  rate=self.audio_file.getframerate(),
                                  output=True)
        self.chunk_size = chunk_size

        ### Audio Contoll Variables
        self.paused_position = 0        # Position in which we paused
        self.audio_playing = False      # Flag of Playing Auduo
        self.paused_for_input = False   # Flag of Pausing for input
        self.initialTime = 0            # intial time of starting our Audio
        self.pausedTime = 0             # Experiment Time in which we paused our stream
        self.nextProbeTime = 0          # Timing of a next probe
        self.initFlag = True            # Flag for initialization
        self.tpDist = tpDistribution

        ### Debug and Return variables
        self.timingLog = []
        self.responsesTiming = []
        self.verbose = verbose
        self.output_control = output_control;
        sys.stdout = self.output_control  # Redirect standard output

        ### Staging and Experiment Mode:
        self.currentStage = currentStage;
        self.nextStage = nextStage #What Kind of stage do you expect next  
        self.tpTimingMode = tpTimingMode# When to Display mode:
                                         # + semirandom - With Uniform Distribution, after +/- 60s 
                                         # + aftertrial - After Each Period of listening...
        self.storyPartTiming = storyPartDict['partTimes'] #List of timings of story parts (In seconds)
        self.storyParts = storyPartDict['partNames']      #Names of the story part (for Logging Purposes)
        self.currentStoryPart = 0;

    def log_part_beg(self):

        message = f"\n   Beginning Part: {self.storyParts[self.currentStoryPart]:12} at exp Time {time.perf_counter()- self.initialTime:9.3f}; Duration({self.storyPartTiming[self.currentStoryPart]})";
        self.output_control.write(message)

        self.output_control.writeToEyeLink(message=f"PART {self.storyParts[self.currentStoryPart]:12} BEG")

    def log_part_end(self):

        message = f"   Part ended:     {self.storyParts[self.currentStoryPart]:12} at exp Time {time.perf_counter()- self.initialTime:9.3f}; Duration({self.storyPartTiming[self.currentStoryPart]})"
        self.output_control.write(message)

        self.output_control.writeToEyeLink(message=f"PART {self.storyParts[self.currentStoryPart]:12} END")
        # Write to Psychopy
    def display_fixation_cross(self):
        """
        Displays a '+' symbol in the center of the screen.
        """
        self.screen.fill((255 / 2, 255 / 2, 255 / 2))  # Gray background
        text_surface = self.font.render("+", True, (255, 255, 255))  # White "+"
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()

    def display_thought_probe(self):
        """
        Displays a thought probe question.
        """
        self.screen.fill((255 / 2, 255 / 2, 255 / 2))  # Gray background
        text_surface = self.font.render("Czy byłeś skupiony na historii?", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()

    def play_audio(self):
        """
        Starts/resumes playing the audio from the current position.
        """
        self.audio_playing = True
        self.audio_file.setpos(self.paused_position)  # Resume from the last paused position

        def stream_audio():
            start_time = time.perf_counter()  # Capture the actual start time

            while self.audio_playing:
                data = self.audio_file.readframes(self.chunk_size)
                if data == b'':  # End of file
                    self.audio_playing = False
                    break
                self.stream.write(data)
                # Track the current position (in samples)
                self.paused_position = self.audio_file.tell()
            actual_duration = time.perf_counter() - start_time  # Calculate actual time taken
            self.output_control.write(f"Actual playback duration: {actual_duration:.3f} sec")

        # Play audio on a separate thread
        threading.Thread(target=stream_audio, daemon=True).start()

    def pause_audio(self):
        """
        Pauses the audio by stopping the playback and saving the position.
        """
        self.audio_playing = False  # Stop the current playback thread

    def initialize_audio(self):
        """
        Initializes audio playback and timers.
        """
        if self.initFlag:
            self.initialTime = time.perf_counter()  # Start the timer
            self.pausedTime = self.initialTime
            self.play_audio()

            self.initFlag = False

            if self.tpTimingMode == 'semirandom':
                self.nextProbeTime = uniform(*self.tpDist)  # Initial random interval - FOR SEMI-RANDOM INPUT
            elif self.tpTimingMode == 'aftertrial':
                self.nextProbeTime = self.storyPartTiming[self.currentStoryPart]; 

            if self.verbose:
                self.output_control.write(f"   Initial Probe Time:      {self.nextProbeTime:.3f}; Story Length: {self.audio_file.getnframes()}")
            self.log_part_beg() #Start with the first Trial! Logging
    def run(self):
        """
        Runs the core of our audio trial.
        """
        self.initialize_audio()

        if self.paused_for_input:
            self.display_thought_probe()
        else:
            self.display_fixation_cross()

        ### Thought Probe Initalisation
        if time.perf_counter() - self.pausedTime >= self.nextProbeTime and not self.paused_for_input:  # Check if the probe interval has been reached
            self.output_control.write(f"{time.perf_counter()}  - {self.pausedTime}  >= {self.nextProbeTime}")

            self.probeOnset = time.perf_counter()
            self.pause_audio()
            self.paused_for_input = True
            if self.verbose:
                self.output_control.write(f"   Current Random Interval: {self.nextProbeTime:.3f}; Thought Probe at: {(time.perf_counter() - self.initialTime):.3f}")

                                #Event Name             #Onset              #RT      #Next TP
            self.timingLog.append(["PROBE",(time.perf_counter() - self.initialTime),0,self.nextProbeTime])
            self.responsesTiming.append(("PROBE", (time.perf_counter() - self.initialTime)))
            if self.tpTimingMode == 'aftertrial': # Log the Beginning of the Part Here 
                self.log_part_end()
                self.currentStoryPart +=1# Moving to the next Part



        ### Checking Continously for new story beginning
        if self.tpTimingMode == 'semirandom': # If our mode is semirandom - indpendently check for parts (WORK IN PROGRESS)
            if time.perf_counter() - self.pausedTime >= self.storyPartTiming[self.currentStoryPart]:
                self.log_part_end()
                self.log_part_beg()
                self.currentStoryPart +=1;

        ### Waiting for KeyPress on Thought Probe
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.stream.stop_stream()
                self.stream.close()
                return "exit"

            if self.paused_for_input and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a or event.key == pygame.K_d:
                    if event.key == pygame.K_a:
                        keyname = "A"
                    else:
                        keyname = "D"
                    self.paused_for_input = False
                    self.pausedTime = time.perf_counter()  # Reset timer

                    if self.tpTimingMode == 'semirandom':
                        self.nextProbeTime = uniform(*self.tpDist)  # Initial random interval - FOR SEMI-RANDOM INPUT
                    elif self.tpTimingMode == 'aftertrial':
                        if self.verbose:
                            self.output_control.write(f"   Key {event.key} pressed at: {self.pausedTime - self.initialTime:.3f}; Reaction Time: {(self.pausedTime - self.probeOnset):.3f}")
                        self.output_control.writeToEyeLink(message=f"KEYPRESS {event.key} END")
                        self.log_part_beg()
                        self.nextProbeTime = self.storyPartTiming[self.currentStoryPart]; 
                                           #Event Name                #Onset                         #RT                           #Next TP
                    self.timingLog.append(["KEYPRESS",self.pausedTime  - self.initialTime ,(self.pausedTime - self.probeOnset),self.nextProbeTime])
                    self.responsesTiming.append((keyname , self.pausedTime  - self.initialTime))
                    self.play_audio()

        # Check if the audio has finished
        if self.audio_file.tell() == self.audio_file.getnframes():
            self.output_control.write(f"{time.perf_counter()}  - {self.pausedTime}  >= {self.nextProbeTime}")
            self.stream.stop_stream()
            self.stream.close()
            self.output_control.write("\nStory Ends.")
            return self.nextStage  # End the procedure

        return  self.currentStage # Usually return story1 if audio has not ended or we hadnt exit

# A list of Timings of story 1: Used to Clock the Onset of Each Fragment: (in seconds)
storyTimeDict2 = data = {
    'partNames': ['INTRODUCTION', 'ADAM_1', 'MARIA_1', 'ADAM_2', 'MARIA_2', 'ADAM_3', 'MARIA_3', 'ADAM_4', 'MARIA_4', 'ADAM_5', 'MARIA_5', 'ADAM_6', 'MARIA_6', 'ADAM_7', 
                  'MARIA_7', 'ADAM_8', 'MARIA_8', 'ADAM_9', 'MARIA_9', 'ADAM_10', 'MARIA_10', 'ADAM_11', 'MARIA_11', 'ENDING'],
    'partTimes': [40.505, 56.678, 56.688, 64.743, 59.355, 58.085, 56.379, 64.487, 62.353, 
                  61.563, 57.115, 59.493, 57.019, 56.283, 59.941, 57.679, 58.043, 59.707, 60.827, 59.663, 59.109, 60.443, 58.832, 64.433]
}

storyTimeDict1 = {
    'partNames': ['INTRODUCTION', 'KASIA_1', 'JANEK_1', 'KASIA_2', 'JANEK_2', 'KASIA_3', 'JANEK_3', 'KASIA_4', 'JANEK_4', 'KASIA_5', 'JANEK_5', 'KASIA_6', 'JANEK_6', 'KASIA_7', 
                  'JANEK_7', 'KASIA_8', 'JANEK_8', 'KASIA_9', 'JANEK_9', 'KASIA_10', 'JANEK_10',  'KASIA_11', 'JANEK_11', 'ENDING'],
    'partTimes': [35.257, 61.296, 59.525, 56.655, 59.92, 65.147, 57.574, 61.99, 65.329, 60.965, 
                  60.454, 57.947, 64.176, 60.955, 60.241, 59.472, 57.115, 59.387, 57.306, 58.693, 62.267, 58.063, 61.295, 55.909]
}
